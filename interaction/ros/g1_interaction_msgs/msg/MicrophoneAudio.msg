# Frame axes: x forward, y left, z up. Samples interleave channels by time.
std_msgs/Header header
uint32 sample_rate
uint8 channel_count
float32[] samples
