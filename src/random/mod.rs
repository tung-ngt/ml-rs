use std::f32::consts::PI;

pub mod pcg;

pub trait RandomGenerator {
    fn seed(&mut self, seed: u64);
    fn next_u32(&mut self) -> u32;
    fn next_f32(&mut self) -> f32;
    fn next_normal(&mut self, mean: f32, std_dev: f32) -> f32 {
        let u1 = self.next_f32().max(f32::MIN_POSITIVE);
        let u2 = self.next_f32();

        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + z0 * std_dev
    }
    fn next_uniform(&mut self, start: f32, end: f32) -> f32 {
        let range = end - start;
        self.next_f32() * range + start
    }

    fn next_shuffle(&mut self, no_elements: usize) -> Vec<usize> {
        let mut elements: Vec<_> = (0..no_elements).collect();
        let mut shuffled = Vec::with_capacity(no_elements);

        while !elements.is_empty() {
            let no_elements_left = elements.len();
            let i = self.next_u32() as usize % no_elements_left;
            shuffled.push(elements.swap_remove(i));
        }

        shuffled
    }
}
