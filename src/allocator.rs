use std::collections::{BTreeMap, VecDeque};
pub(crate) const ALIGNMENT: u64 = wgpu::COPY_BUFFER_ALIGNMENT;

pub type Offset = u64;
pub type PhysicalSize = u64;
pub type VirtualSize = u64;
pub type EmptySize = u64;
// old offset, new offset, size
pub type RelocationList = VecDeque<(Offset, Offset, PhysicalSize)>;

#[derive(Debug)]
pub struct UsedBytes {
    // physical size means the actual size in memory
    pub(crate) physical_size: u64,
    // virtual means the memory occupied by the stored data
    pub(crate) _virtual_size: u64,
}

/// Pads a byte slice to be a multiple of `wgpu::COPY_BUFFER_ALIGNMENT`.
fn pad_data(data: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    let align = ALIGNMENT as usize;
    let unpadded_len = data.len();
    let padded_len = (unpadded_len + align - 1) & !(align - 1);
    if unpadded_len == padded_len {
        return std::borrow::Cow::Borrowed(data);
    }
    let mut padded_data = data.to_vec();
    padded_data.resize(padded_len, 0);
    std::borrow::Cow::Owned(padded_data)
}

pub struct SSBOAllocator {
    // mock buffer
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) buffer_size: u64,
    pub(crate) used_size: u64,
    // free blocks by size: for finding a suitable block (best-fit)
    pub(crate) free_by_size: BTreeMap<EmptySize, VecDeque<Offset>>,
    // free blocks by offset: for coalescing
    pub(crate) free_by_offset: BTreeMap<Offset, EmptySize>,
    // key offset, value size
    pub(crate) allocated_blocks: BTreeMap<Offset, UsedBytes>,
}

impl SSBOAllocator {
    pub fn new(device: &wgpu::Device, name: &str, size: u64) -> Self {
        let memory = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut free_by_size: BTreeMap<EmptySize, VecDeque<Offset>> = BTreeMap::new();
        let mut free_by_offset: BTreeMap<Offset, EmptySize> = BTreeMap::new();
        let mut deque = VecDeque::new();
        deque.push_back(0);
        free_by_size.insert(size, deque);
        free_by_offset.insert(0, size);

        Self {
            buffer: memory,
            buffer_size: size,
            used_size: 0,
            free_by_size,
            free_by_offset,
            allocated_blocks: BTreeMap::new(),
        }
    }

    pub fn get_size(&self) -> u64 {
        self.buffer_size
    }

    pub fn get_used_size(&self) -> u64 {
        self.used_size
    }

    pub fn get_free_size(&self) -> u64 {
        self.buffer_size - self.used_size
    }

    pub fn get_allocation_count(&self) -> usize {
        self.allocated_blocks.len()
    }

    pub fn get_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Allocates data on the mock memory, returning the starting offset.
    /// An optional padding can be specified.
    pub fn allocate(&mut self, queue: &wgpu::Queue, data: &[u8], padding: Option<u64>) -> Result<Offset, &'static str> {
        let virtual_size = data.len() as u64;
        let requested_physical_size = virtual_size + padding.unwrap_or(0);

        // Round up the physical size to the required alignment.
        // This is the key to ensuring all future offsets are aligned.
        let physical_size = (requested_physical_size + ALIGNMENT - 1) & !(ALIGNMENT - 1);

        // find a suitable free block (best-fit)
        if let Some((&block_size, offsets)) = self.free_by_size.range_mut(physical_size..).next() {
            // return early if block too small
            if block_size < physical_size {
                return Err("Block size too small");
            }

            // get an offset from the list of blocks of this size
            if let Some(offset) = offsets.pop_front() {
                // this block is no longer free, remove it from tracking maps
                self.free_by_offset.remove(&offset);
                if offsets.is_empty() {
                    // if this was the last block of this size, remove the size entry entirely
                    self.free_by_size.remove(&block_size);
                }

                // add the new block to our allocated list
                self.allocated_blocks.insert(offset, 
                    UsedBytes {
                        physical_size,
                        _virtual_size: virtual_size,
                    });

                self.used_size += physical_size;

                // copy the data into the mock memory
                let padded_data = pad_data(data);
                queue.write_buffer(&self.buffer, offset, &padded_data);

                // if the block we used was larger than needed,
                // add the leftover piece back to the free lists
                if block_size > physical_size {
                    let remainder_offset = offset + physical_size;
                    let remainder_size = block_size - physical_size;
                    self.free_by_offset.insert(remainder_offset, remainder_size);
                    self.free_by_size
                        .entry(remainder_size)
                        .or_default()
                        .push_back(remainder_offset);
                }
                return Ok(offset);
            } else {
                return Err("No suitable offset found");
            }
        } else {
            return Err("No suitable block found");
        }
    }

    // Modifies data in the buffer, fails if data size is too big.
    pub fn modify(&mut self, queue: &wgpu::Queue, offset: Offset, data: &[u8]) -> Result<(), &'static str> {
        if let Some(size) = self.allocated_blocks.get(&offset) {
            if data.len() as u64 <= size.physical_size {
                let padded_data = pad_data(data);
                queue.write_buffer(&self.buffer, offset, &padded_data);
                return Ok(());
            } else {
                return Err("Data size too big for buffer");
            }
        } else {
            return Err("Offset not allocated");
        }
    }
    
    // Deallocates data on the mock memory and deletes the data from the buffer.
    pub fn deallocate_wipe(&mut self, queue: &wgpu::Queue, offset: Offset) -> Result<(), &'static str> {
        let mut current_size = if let Some(size) = self.allocated_blocks.remove(&offset) {
            let zeroed_data = vec![0u8; size.physical_size as usize];
            queue.write_buffer(&self.buffer, offset, &pad_data(&zeroed_data));
            self.used_size -= size.physical_size;
            size
        } else {
            // this offset was not allocated, nothing to do
            return Err("Offset wasn't allocated, can't deallocate");
        };
        let mut current_offset = offset;

        // check for a free block starting exactly where the current block ends
        let right_neighbor_offset = current_offset + current_size.physical_size;
        if let Some(right_neighbor_size) = self.free_by_offset.remove(&right_neighbor_offset) {
            // remove old entry from the size-based map
            if let Some(offsets) = self.free_by_size.get_mut(&right_neighbor_size) {
                offsets.retain(|&o| o != right_neighbor_offset);
                if offsets.is_empty() {
                    self.free_by_size.remove(&right_neighbor_size);
                }
            }
            // add its size to the current block.
            current_size.physical_size += right_neighbor_size;
        }

        // we look for a free block that ends exactly where our current block begins.
        if let Some((&left_offset, &left_size)) = self.free_by_offset.range(..current_offset).next_back() {
            if left_offset + left_size == current_offset {
                // remove old entries from both free maps.
                self.free_by_offset.remove(&left_offset);
                if let Some(offsets) = self.free_by_size.get_mut(&left_size) {
                    offsets.retain(|&o| o != left_offset);
                    if offsets.is_empty() {
                        self.free_by_size.remove(&left_size);
                    }
                }
                // update current block's offset and size
                current_offset = left_offset;
                current_size.physical_size += left_size;
            }
        }

        // add the new, potentially larger, free block back to the free space maps.
        self.free_by_offset.insert(current_offset, current_size.physical_size);
        self.free_by_size.entry(current_size.physical_size).or_default().push_back(current_offset);

        Ok(())
    }

    // Returns a list that maps the current offset to the new offset (relocated - moved left by the available empty space); first step of compacting
    pub fn get_relocation_list(&self) -> RelocationList {
        let mut relocation_list = RelocationList::new();

        let mut next_destination_offset: Offset = 0; 
        for (current_source_offset, current_block) in self.allocated_blocks.iter() {
            if *current_source_offset != next_destination_offset {
                relocation_list.push_back((*current_source_offset, next_destination_offset, current_block.physical_size));
            }

            next_destination_offset += current_block.physical_size;
        }

        relocation_list
    }

    // Submits buffer copies to the encoder; used for compacting, and is the second step of the process
    pub fn submit_copies_to_encoder(
        &mut self, 
        encoder: &mut wgpu::CommandEncoder, 
        relocation_list: &mut RelocationList
    ) {
        while let Some((old_offset, new_offset, size)) = relocation_list.pop_front() {
            // filter out no-ops where old_offset == new_offset
            if old_offset != new_offset && size > 0 {
                encoder.copy_buffer_to_buffer(
                    &self.buffer, 
                    old_offset, 
                    &self.buffer, 
                    new_offset, 
                    size as wgpu::BufferAddress
                );
            }
        }
    }

    // Updates the allocator schema with the relocation list, is the final step of compacting
    pub fn update_allocator_schema_with_relocation_list(&mut self, relocation_list: &RelocationList) {
        let relocation_map: BTreeMap<Offset, Offset> = relocation_list
            .iter()
            .map(|(old, new, _)| (*old, *new))
            .collect();

        for (old_offset, new_offset) in relocation_map.iter() {
            let used_bytes = self.allocated_blocks
                .remove(old_offset)
                .expect("Error: Allocated block not found at old_offset during relocation update.");
            
            self.allocated_blocks.insert(*new_offset, used_bytes);
        }

        let used_size = self.allocated_blocks
            .values()
            .map(|ub| ub.physical_size)
            .sum::<Offset>();

        let new_free_offset = used_size;
        let new_free_size = self.buffer_size - used_size;

        self.free_by_offset.clear();
        self.free_by_size.clear();

        if new_free_size > 0 {
            self.free_by_offset.insert(new_free_offset, new_free_size);
            self.free_by_size.entry(new_free_size).or_default().push_front(new_free_offset);
        }
    }
}
