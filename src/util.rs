use crate::allocator::{PhysicalSize, ALIGNMENT};

pub(crate) async fn setup_wgpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .unwrap();

    (device, queue)
}

pub(crate) async fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    size: PhysicalSize,
) -> Vec<u8> {
    let aligned_size = (size + ALIGNMENT - 1) & !(ALIGNMENT - 1);
    
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: aligned_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, aligned_size);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device.poll(wgpu::wgt::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::MAX) }).expect("Polling failed");
    rx.receive().await.unwrap().unwrap();

    let data = buffer_slice.get_mapped_range().to_vec();
    staging_buffer.unmap();
    
    // Return only the originally requested size, not the padded data.
    data[..size as usize].to_vec()
}