import numpy as np


class MetadataHandler:
    def __call__(self, *args, **kwds):
        self.shift_bits = 40  # Number of bits to shift for runNumber

    def generate_global_id(self, run_number, event_number):
        """
        Combines runNumber and eventNumber into a unique 64-bit integer.
        Uses a 40-bit shift to accommodate very large event numbers.
        """
        # Ensure we are using 64-bit integers to prevent overflow in Python/NumPy
        global_id = (np.uint64(run_number) << self.shift_bits) | np.uint64(event_number)
        return int(global_id)

    def recover_from_id(self, global_id):
        """
        Reverses the bit-shift to recover the original run and event numbers.
        """
        # Recover runNumber by shifting back to the right
        run_number = global_id >> self.shift_bits

        # Recover eventNumber using a bitmask of 40 ones
        # (1 << 40) - 1 creates a binary number with forty 1s
        mask = (1 << self.shift_bits) - 1
        event_number = global_id & mask

        return int(run_number), int(event_number)
