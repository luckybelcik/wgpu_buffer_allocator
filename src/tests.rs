#[cfg(test)]
mod tests {
    use core::panic;
    use rand::Rng;
    use std::time::Instant;

    pub(crate) const BIG_INITIAL_MEMORY_SIZE: u64 = 1_048_576;
    pub(crate) const INITIAL_MEMORY_SIZE: u64 = 1024;

    use crate::{allocator::*, util::{read_buffer, setup_wgpu}};

    #[test]
    fn test_simple_allocation_and_deallocation() {
        pollster::block_on(async {
            let (device, queue) = setup_wgpu().await;
            let mut allocator = SSBOAllocator::new(&device, "test", INITIAL_MEMORY_SIZE);
            let data = vec![1; 100];

            // Allocate
            let offset = allocator.allocate(&queue, &data, None).expect("Allocation failed");
            assert_eq!(offset, 0);
            if let Some(size) = allocator.allocated_blocks.get(&0) {
                assert_eq!(size.physical_size, 100);
            } else {
                panic!("Allocated block not found");
            }
            assert_eq!(allocator.free_by_offset.get(&100), Some(&(INITIAL_MEMORY_SIZE - 100)));

            // Deallocate
            allocator.deallocate_wipe(&queue, offset).expect("Deallocation failed");
            assert!(allocator.allocated_blocks.is_empty());
            let buffer_data = read_buffer(&device, &queue, &allocator.buffer, 100).await;
            assert_eq!(buffer_data.iter().sum::<u8>(), 0);
            assert_eq!(allocator.free_by_offset.get(&0), Some(&INITIAL_MEMORY_SIZE));
            assert_eq!(allocator.free_by_size.get(&INITIAL_MEMORY_SIZE).unwrap().len(), 1);
        });
    }

    #[test]
    fn test_allocation_fails() {
        pollster::block_on(async {
            let (device, queue) = setup_wgpu().await;
            let mut allocator = SSBOAllocator::new(&device, "test_buffer", INITIAL_MEMORY_SIZE);
            let data = vec![1; allocator.buffer_size as usize + 1];

            // Allocate
            if allocator.allocate(&queue, &data, None).is_ok() {
                panic!("Modification should have failed");
            }
        });
    }

    #[test]
    fn test_simple_allocation_and_deallocation_padded() {
        pollster::block_on(async {
            let (device, queue) = setup_wgpu().await;
            let mut allocator = SSBOAllocator::new(&device, "test_buffer", INITIAL_MEMORY_SIZE);
            let data = vec![1; 90];

            // Allocate
            // 90 + 10 = 100, which is already aligned.
            let offset = allocator.allocate(&queue, &data, Some(10)).expect("Allocation failed");
            assert_eq!(offset, 0);
            if let Some(size) = allocator.allocated_blocks.get(&0) {
                assert_eq!(size.physical_size, 100);
                assert_eq!(size._virtual_size, 90);
            } else {
                panic!("Allocated block not found");
            }
            assert_eq!(allocator.free_by_offset.get(&100), Some(&(INITIAL_MEMORY_SIZE - 100)));

            // Deallocate
            allocator.deallocate_wipe(&queue, offset).expect("Deallocation failed");
            assert!(allocator.allocated_blocks.is_empty());
            let buffer_data = read_buffer(&device, &queue, &allocator.buffer, 100).await;
            assert_eq!(buffer_data.iter().sum::<u8>(), 0);
            assert_eq!(allocator.free_by_offset.get(&0), Some(&INITIAL_MEMORY_SIZE));
            assert_eq!(allocator.free_by_size.get(&INITIAL_MEMORY_SIZE).unwrap().len(), 1);
        });
    }

    #[test]
    fn test_modify() {
        pollster::block_on(async {
            let (device, queue) = setup_wgpu().await;
            let mut allocator = SSBOAllocator::new(&device, "test_buffer", INITIAL_MEMORY_SIZE);
            let data = vec![1; 100];

            // Allocate
            let offset = allocator.allocate(&queue, &data, None).expect("Allocation failed");
            assert_eq!(offset, 0);

            // Modify
            let new_data = vec![2; 50];
            allocator.modify(&queue, offset, &new_data).expect("Modification failed");
            let buffer_data = read_buffer(&device, &queue, &allocator.buffer, 100).await;
            assert_eq!(buffer_data[0..50], [2; 50]);
        });
    }

    #[test]
    fn test_modify_fails() {
        pollster::block_on(async {
            let (device, queue) = setup_wgpu().await;
            let mut allocator = SSBOAllocator::new(&device, "test_buffer", INITIAL_MEMORY_SIZE);
            let data = vec![1; 100];

            // Allocate
            let offset = allocator.allocate(&queue, &data, None).expect("Allocation failed");
            assert_eq!(offset, 0);

            // Modify
            let new_data = vec![2; 101];
            if allocator.modify(&queue, offset, &new_data).is_ok() {
                panic!("Modification should have failed");
            }
        });
    }

    #[test]
    fn test_best_fit_and_split() {
        pollster::block_on(async {
            let (device, queue) = setup_wgpu().await;
            let mut allocator = SSBOAllocator::new(&device, "test_buffer", INITIAL_MEMORY_SIZE);
            // Create two free blocks: one of size 24, one of size 1000
            allocator.free_by_size.clear();
            allocator.free_by_offset.clear();
            allocator.free_by_offset.insert(0, 24);
            allocator.free_by_size.entry(24).or_default().push_back(0);
            allocator.free_by_offset.insert(24, 1000);
            allocator.free_by_size.entry(1000).or_default().push_back(24);

            // Allocate 18 bytes. This will be aligned up to 20.
            // It should pick the 24-byte block (best-fit)
            let data = vec![1; 18];
            let offset = allocator.allocate(&queue, &data, None).expect("Allocation failed");

            assert_eq!(offset, 0); // Used the first block
            if let Some(size) = allocator.allocated_blocks.get(&0) {
                assert_eq!(size.physical_size, 20); // Aligned size
            } else {
                panic!("Allocated block not found");
            }
            // Check that the remainder of the split block is now a new free block
            assert_eq!(allocator.free_by_offset.get(&20), Some(&4)); // 24 - 20 = 4
            assert_eq!(allocator.free_by_size.get(&4).unwrap().front(), Some(&20));
            // The 1000-byte block should be untouched
            assert_eq!(allocator.free_by_offset.get(&24), Some(&1000));
        });
    }

    #[test]
    fn test_coalesce_right() {
        pollster::block_on(async {
            let (device, queue) = setup_wgpu().await;
            let mut allocator = SSBOAllocator::new(&device, "test_buffer", INITIAL_MEMORY_SIZE);
            let data1 = vec![1; 10]; // Aligns to 12
            let data2 = vec![2; 20]; // Aligns to 20

            let offset1 = allocator.allocate(&queue, &data1, None).unwrap(); // 0-12
            let offset2 = allocator.allocate(&queue, &data2, None).unwrap(); // 12-32

            allocator.deallocate_wipe(&queue, offset1).expect("Deallocation failed"); // Frees 0-10
            assert_eq!(allocator.free_by_offset.get(&0), Some(&12));

            allocator.deallocate_wipe(&queue, offset2).expect("Deallocation failed"); // Frees 10-30, should merge with 0-10 and all the space to the right
            assert!(allocator.allocated_blocks.is_empty());
            assert_eq!(allocator.free_by_offset.get(&0), Some(&INITIAL_MEMORY_SIZE)); // Merged block
            assert_eq!(allocator.free_by_offset.len(), 1); // Merged block + rest of memory
        });
    }

    #[test]
    fn test_coalesce_left() {
        pollster::block_on(async {
            let (device, queue) = setup_wgpu().await;
            let mut allocator = SSBOAllocator::new(&device, "test_buffer", INITIAL_MEMORY_SIZE);
            let data1 = vec![1; 10]; // Aligns to 12
            let data2 = vec![2; 20]; // Aligns to 20

            let offset1 = allocator.allocate(&queue, &data1, None).unwrap(); // 0-12
            let offset2 = allocator.allocate(&queue, &data2, None).unwrap(); // 12-32

            allocator.deallocate_wipe(&queue, offset2).expect("Deallocation failed"); // Frees 10-30, merges to the right until the end
            assert_eq!(allocator.free_by_offset.get(&12), Some(&(INITIAL_MEMORY_SIZE - 12)));

            allocator.deallocate_wipe(&queue, offset1).expect("Deallocation failed"); // Frees 0-10, should merge with 10-30
            assert!(allocator.allocated_blocks.is_empty());
            assert_eq!(allocator.free_by_offset.get(&0), Some(&INITIAL_MEMORY_SIZE)); // Merged block
        });
    }

    #[test]
    fn test_coalesce_both_sides() {
        pollster::block_on(async {
            let (device, queue) = setup_wgpu().await;
            let mut allocator = SSBOAllocator::new(&device, "test_buffer", INITIAL_MEMORY_SIZE);
            let data1 = vec![1; 10]; // aligns to 12
            let data2 = vec![2; 20]; // aligns to 20
            let data3 = vec![3; 30]; // aligns to 32

            let offset1 = allocator.allocate(&queue, &data1, None).unwrap();
            let offset2 = allocator.allocate(&queue, &data2, None).unwrap();
            let offset3 = allocator.allocate(&queue, &data3, None).unwrap();

            allocator.deallocate_wipe(&queue, offset1).expect("Deallocation failed"); // Free block at 0, size 12
            allocator.deallocate_wipe(&queue, offset3).expect("Deallocation failed"); // Free block at 32

            // At this point, we have: free(0,12), allocated(12,20), free(32, ...)
            assert_eq!(allocator.free_by_offset.get(&0), Some(&12));
            assert_eq!(allocator.free_by_offset.get(&32), Some(&(INITIAL_MEMORY_SIZE - 32)));

            allocator.deallocate_wipe(&queue, offset2).expect("Deallocation failed"); // Deallocate middle block

            // Should merge all three into one big block
            assert_eq!(allocator.free_by_offset.get(&0), Some(&INITIAL_MEMORY_SIZE));
        });
    }

    #[test]
    fn test_compaction_no_moves_needed() {
        pollster::block_on(async {
            let (device, queue) = setup_wgpu().await;
            let mut allocator = SSBOAllocator::new(&device, "test_buffer", INITIAL_MEMORY_SIZE);
            let data1 = vec![1; 100]; // aligns to 12
            let data2 = vec![2; 200]; // aligns to 20
            let data3 = vec![3; 300]; // aligns to 32

            allocator.allocate(&queue, &data1, None).unwrap();
            allocator.allocate(&queue, &data2, None).unwrap();
            allocator.allocate(&queue, &data3, None).unwrap();

            let relocation_list = allocator.get_relocation_list();
            assert!(relocation_list.is_empty());
        });
    }

    #[test]
    fn test_simple_fragmentation() {
        pollster::block_on(async {
            let (device, queue) = setup_wgpu().await;
            let mut allocator = SSBOAllocator::new(&device, "test_buffer", INITIAL_MEMORY_SIZE);
            let data1 = vec![1; 100];
            let data2 = vec![2; 200];
            let data3 = vec![3; 300];

            allocator.allocate(&queue, &data1, None).expect("Allocation failed");
            let offset2 = allocator.allocate(&queue, &data2, None).expect("Allocation failed");
            allocator.allocate(&queue, &data3, None).expect("Allocation failed");

            allocator.deallocate_wipe(&queue, offset2).expect("Deallocation failed");

            let relocation_list = allocator.get_relocation_list();
            assert_eq!(relocation_list.len(), 1);
            assert_eq!(relocation_list[0], (300, 100, 300));
        });
    }

    #[test]
    fn test_stress_and_fragmentation() {
        pollster::block_on(async {
            let (device, queue) = setup_wgpu().await;
            let mut allocator = SSBOAllocator::new(&device, "test_buffer", BIG_INITIAL_MEMORY_SIZE);
            let mut rng = rand::rng();
            let mut active_allocations: Vec<Offset> = Vec::new();
            let mut successful_operations: usize = 0;
            let mut failed_operations: usize = 0;
            let mut successful_allocations: usize = 0;
            let mut failed_allocations: usize = 0;
            let mut successful_deallocations: usize = 0;
            let mut failed_deallocations: usize = 0;
            let mut successful_modifications: usize = 0;
            let mut failed_modifications: usize = 0;

            const NUM_OPERATIONS: usize = 300;
            const MAX_ALLOC_SIZE: usize = 50_000;
            const MAX_PADDING: u64 = 32;

            let start_time = Instant::now();

            for _ in 0..NUM_OPERATIONS {
                let op = rng.random_range(0..100);

                match op {
                    0..=60 => {
                        let alloc_size = rng.random_range(1..=MAX_ALLOC_SIZE);
                        let data: Vec<u8> = (0..alloc_size).map(|_| rng.random()).collect();

                        if rng.random_bool(0.2) {
                            let padding: u64 = rng.random_range(0..=MAX_PADDING) as u64;
                            if let Ok(offset) = allocator.allocate(&queue, &data, Some(padding)) {
                                successful_operations += 1;
                                successful_allocations += 1;
                                active_allocations.push(offset);
                            } else {
                                failed_allocations += 1;
                                failed_operations += 1;
                            }
                        } else {
                            if let Ok(offset) = allocator.allocate(&queue, &data, None) {
                                successful_operations += 1;
                                successful_allocations += 1;
                                active_allocations.push(offset);
                            } else {
                                failed_allocations += 1;
                                failed_operations += 1;
                            }
                        }
                    }
                    61..=90 => {
                        if !active_allocations.is_empty() {
                            let idx_to_remove = rng.random_range(0..active_allocations.len());
                            let offset = active_allocations.swap_remove(idx_to_remove);
                            if allocator.deallocate_wipe(&queue, offset).is_ok() {
                                successful_operations += 1;
                                successful_deallocations += 1;
                            } else {
                                failed_operations += 1;
                                failed_deallocations += 1;
                            }
                        }
                    }
                    _ => {
                        if let Some(&offset) = active_allocations.get(rng.random_range(0..(active_allocations.len()).max(1)).max(1).max(1)) {
                            if let Some(used_bytes) = allocator.allocated_blocks.get(&offset) {
                                let new_size = rng.random_range(1..=used_bytes.physical_size);
                                let new_data: Vec<u8> = (0..new_size).map(|_| rng.random()).collect();
                                if let Ok(_) = allocator.modify(&queue, offset, &new_data) {
                                    successful_operations += 1;
                                    successful_modifications += 1;
                                } else {
                                    failed_operations += 1;
                                    failed_modifications += 1;
                                }
                            }
                        }
                    }
                }
            }

            let compaction_start_time = Instant::now();

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Encoder"),
            });

            let mut relocation_list = allocator.get_relocation_list();
            allocator.submit_copies_to_encoder(&mut encoder, &mut relocation_list);
            allocator.update_allocator_schema_with_relocation_list(&relocation_list);

            queue.submit(Some(encoder.finish()));

            let duration = start_time.elapsed();
            let duration_compaction = compaction_start_time.elapsed();

            let total_allocated_physical: u64 = allocator.allocated_blocks.values().map(|ub| ub.physical_size).sum();
            let total_allocated_virtual: u64 = allocator.allocated_blocks.values().map(|ub| ub._virtual_size).sum();

            let internal_fragmentation = total_allocated_physical - total_allocated_virtual;

            let total_free_space: u64 = allocator.free_by_offset.values().sum();
            let largest_free_block = allocator.free_by_size.keys().next_back().copied().unwrap_or(0);
            
            let external_fragmentation_ratio = if total_free_space > 0 {
                1.0 - (largest_free_block as f64 / total_free_space as f64)
            } else {
                0.0
            };

            println!("\n--- Allocator Stress Test Results ---");
            println!("Total operations: {}", NUM_OPERATIONS);
            println!("Time elapsed: {:?}", duration);
            println!("Compaction time elapsed: {:?}", duration_compaction);
            println!("Operations per second: {:.2}", NUM_OPERATIONS as f64 / duration.as_secs_f64());
            println!("\n--- Operation Information ---");
            println!("Successful operations: {}", successful_operations);
            println!("Failed operations: {}", failed_operations);
            println!("Successful alloc: {}", successful_allocations);
            println!("Failed alloc: {}", failed_allocations);
            println!("Successful dealloc: {}", successful_deallocations);
            println!("Failed dealloc: {}", failed_deallocations);
            println!("Successful modify: {}", successful_modifications);
            println!("Failed modify: {}", failed_modifications);
            println!("\n--- Memory State After Test ---");
            println!("Active allocations: {}", allocator.allocated_blocks.len());
            println!("Total physical memory allocated: {} bytes", total_allocated_physical);
            println!("Total virtual memory allocated: {} bytes", total_allocated_virtual);
            println!("Internal fragmentation (padding): {} bytes", internal_fragmentation);
            println!("Total free space: {} bytes", total_free_space);
            println!("Number of free blocks: {}", allocator.free_by_offset.len());
            println!("Largest free block: {} bytes", largest_free_block);
            println!("External fragmentation ratio: {:.4}", external_fragmentation_ratio);
            println!("-------------------------------------\n");
        });
    }
}