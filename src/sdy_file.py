import numpy as np

class SDYFile:
    TOTAL_CHANNELS = 44 + 1079

    details_list = ['last_name',
                    'first_name',
                    'middle_initial',
                    'gender',
                    'patient_id',
                    'physcian',
                    'date_of_birth',
                    'procedure',
                    'procedure_id',
                    'accession',
                    'ffr',
                    'ffr_suid',
                    'refering_physcian',
                    'additional_pt_history',
                    'ivus_suid',
                    'department',
                    'institution',
                    'cathlab_id']

    chan_start = np.array([0])  # Just 3560
    chan_unknown1 = np.array([1])  # Just 96
    chan_blank1 = np.array([2, 3])  # Just 0
    chan_unknown2 = np.array([4, 6, 8, 10])
    chan_pd = np.array([5, 7, 9, 11])
    chan_unknown_pa = np.array([12, 13, 14, 15])  # Sometime pa, sometimes noise
    chan_pa = np.array([16, 17, 18, 19])
    chan_unknown3 = np.array([20, 21, 22, 23])  # ?Noise
    chan_ecg = np.array([24, 26, 28, 30])
    chan_unknown4 = np.array([25])  # Unknown ?Marker
    chan_unknown5 = np.array([27])  # Unknown ?Marker
    chan_unknown6 = np.array([29])  # Unknown ?Marker
    chan_unknown7 = np.array([31])  # Unknown ?Marker
    chan_flow = np.array([32, 33])
    chan_flow_average = np.array([34])  # Flow beat-by-beat average
    chan_blank3 = np.array([35])  # Unknown - high
    chan_calc2 = np.array([36, 37])
    chan_calc3 = np.array([38, 39])
    chan_flow_baseline = np.array([40])  # Baseline Fixed 48 or 114
    chan_blank4 = np.array([41])  # Fixed 1
    chan_calc5 = np.array([42])  # Calc
    chan_flow_scale = np.array([43])  # flow scale for spectrum

    # NOTE(guilherme): the number of channels in the arrays above correlate to the
    # frequency, 50hz = 1 channel, 100hz = 2 channels, 150hz = 3 channels, 200hz = 4 channels. 

    # 43 to 87 inclusive - UNUSED - 0

    # 88:344 - Spectrum A
    # 244:600 - Spectrum B

    # 600 to 1111 inclusive - UNUSED - 0

    # 1112 - ? Part of checksum
    # 1113 - Various descrete values
    # 1114 - Like about 15 minutes. Ascending wtith occaional drop down
    # 1115 - EVENT 2
    # 1116 - 0
    # 1117 - 0 [ 1116 + 1117 uint32 SPACE]
    chan_ms_loop = np.array([1118])
    chan_ms_loop_counter = np.array([1119])
    # 1118 - Number of seconds per minute 65535 per minutes
    # 1119 - Number of minutes [1118, 1119 - uint32 - elapsed time stamp in packet]
    # 1120 - Event
    # 1121 - CRC
    # 1122 - SPACE - 0
    # End


    def __init__(self, sdy_fl):

        self.sdy_fl = sdy_fl
        self.plot_order = False

        self.sampling_rate = 50  ###Check this is only a guess

        with open(sdy_fl, 'rb') as sdy_f:
            sdy_f.seek(0)
            self.file_header = sdy_f.read(16)
            self.file_patient = sdy_f.read(512*18)
            self.file_raw_data = sdy_f.read(-1)

        self.file_type = np.frombuffer(self.file_header, offset=0, dtype=np.uint32, count=1)
        self.date_time = np.frombuffer(self.file_header, offset=4, dtype=np.uint32, count=2)
        self.exam_type = np.frombuffer(self.file_header, offset=12, dtype=np.int32, count=1)
        # 3=pressure, 4=Flow, 5=Combo

        pt_details = {}

        for i, detail in enumerate(self.details_list):
            start = i * 512
            end = (i+1) * 512
            pt_details[detail] = self.file_patient[start:end].decode('utf-16').replace('\x00', '').strip()

        self.pt_details = pt_details

        raw_data = np.frombuffer(self.file_raw_data, dtype=np.uint16, count=-1)
        data_width = len(raw_data) // self.TOTAL_CHANNELS
        data_len = data_width * self.TOTAL_CHANNELS
        remainder = len(raw_data) % self.TOTAL_CHANNELS

        # The axis to the right is the one which cycles the fastest through the data
        # Note, as from buffers is not-writable
        self.frame = raw_data[:data_len].reshape((data_width, self.TOTAL_CHANNELS))

        # the indexing by [:, [a,b]] leads to a copy so these are mutable
        self.pa = self.frame[:, self.chan_pa].flatten()
        self.pd = self.frame[:, self.chan_pd].flatten()
        self.ecg = self.frame[:, self.chan_ecg].flatten()

        self.flow = self.frame[:, self.chan_flow].flatten().astype(np.float32)
        self.flow_scale = self.frame[:, self.chan_flow_scale].repeat(2).astype(np.float32)
        self.flow_baseline = self.frame[:, self.chan_flow_baseline].repeat(2).astype(np.float32)
        self.flow_true = 5 * (self.flow-self.flow_baseline) / self.flow_scale

        self.flow_average = self.frame[:, self.chan_flow_average].flatten()
        self.calc2 = self.frame[:, self.chan_calc2].flatten()
        self.calc3 = self.frame[:, self.chan_calc3].flatten()

        self.spectrum = self.frame[:, 88:600].reshape((-1, 256)).T

        self.elapsed_time_ms = np.zeros(data_len, dtype=np.int64)
        self.elapsed_time_ms = self.frame[:, self.chan_ms_loop] + (self.frame[:, self.chan_ms_loop_counter] * 65536)

    def write_file(self, path):
        out_frame = self.frame.copy()
        with open(path, "wb") as sdy_out_f:
            sdy_out_f.write(self.file_header)
            sdy_out_f.write(self.file_patient)
            sdy_out_f.write(out_frame.tobytes())

    def replace_and_write(self, flow, path):

        flow = flow.reshape((-1, 2)).astype(np.uint16)

        out_frame = self.frame.copy()
        out_frame[1000:-1000, 32] = flow[1000:-1000, 0]
        out_frame[1000:-1000, 33] = flow[1000:-1000, 1]

        with open(path, "wb") as sdy_out_f:
            sdy_out_f.write(self.file_header)
            sdy_out_f.write(self.file_patient)
            sdy_out_f.write(out_frame.tobytes())


def main():
    sdy = SDYFile("./sample_data/sample.sdy")

    print("hello")



if __name__ == "__main__":
    main()