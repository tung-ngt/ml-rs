use std::num::Wrapping;

use super::RandomGenerator;

pub struct PCG {
    state: Wrapping<u64>,
    inc: Wrapping<u64>,
}

impl PCG {
    const MULTIPLIER: Wrapping<u64> = Wrapping(6364136223846793005);

    pub fn new(seed: u64, stream: u64) -> Self {
        let mut rng = Self {
            state: Wrapping(0),
            inc: Wrapping((stream << 1) | 1),
        };
        rng.seed(seed);
        rng
    }
}

impl RandomGenerator for PCG {
    fn seed(&mut self, seed: u64) {
        self.state = Wrapping(0);
        self.next_u32();
        self.state += Wrapping(seed);
        self.next_u32();
    }

    fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state * Self::MULTIPLIER + self.inc;

        // XSH-RR transformation: XOR-Shift + Rotate-Right
        let xor_shifted = (((old_state.0 >> 18) ^ old_state.0) >> 27) as u32;
        let rot = (old_state.0 >> 59) as u32; // Top 5 bits determine rotation
        xor_shifted.rotate_right(rot)
    }

    fn next_f32(&mut self) -> f32 {
        let rnd_u32 = self.next_u32();
        (rnd_u32 as f32) * (1.0 / 4294967296.0) // Divide by 2^32
    }
}

#[cfg(test)]
mod pcg_tests {
    use crate::random::RandomGenerator;

    use super::PCG;

    #[test]
    fn uniform_01() {
        let mut gen = PCG::new(42, 1);
        let mut bins = [0; 10];
        let count = 100 * 1000;

        for _ in 0..count {
            match gen.next_f32() {
                x if (0.0..0.1).contains(&x) => bins[0] += 1,
                x if (0.1..0.2).contains(&x) => bins[1] += 1,
                x if (0.2..0.3).contains(&x) => bins[2] += 1,
                x if (0.3..0.4).contains(&x) => bins[3] += 1,
                x if (0.4..0.5).contains(&x) => bins[4] += 1,
                x if (0.5..0.6).contains(&x) => bins[5] += 1,
                x if (0.6..0.7).contains(&x) => bins[6] += 1,
                x if (0.7..0.8).contains(&x) => bins[7] += 1,
                x if (0.8..0.9).contains(&x) => bins[8] += 1,
                x if (0.9..1.0).contains(&x) => bins[9] += 1,
                _ => {}
            }
        }

        for b in bins {
            print!("{} ", (b as f32) / (count as f32))
        }
    }
}
