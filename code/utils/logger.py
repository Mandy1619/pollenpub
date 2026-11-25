'''Copyright (C) 2019 Erik Linder-Norén, Swiss Federal Institute of Technology (ETH Zurich), Matthias Meyer

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Changes to original version: Adopted to work with multi-layer pollen data
'''
import csv
import os

# The original code used the unmaintained `tensorboard_logger` package. We use
# PyTorch's built-in SummaryWriter when tensorboard is installed and always also
# append to a CSV file (<log_dir>/metrics.csv) that can be read without tooling.
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # tensorboard not installed
    SummaryWriter = None


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir) if SummaryWriter is not None else None
        self.csv_path = os.path.join(log_dir, "metrics.csv")

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.list_of_scalars_summary([(tag, value)], step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        new_file = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["step", "tag", "value"])
            for tag, value in tag_value_pairs:
                w.writerow([step, tag, float(value)])
        if self.writer is not None:
            for tag, value in tag_value_pairs:
                self.writer.add_scalar(tag, float(value), step)
            self.writer.flush()
