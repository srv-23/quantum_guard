import pyshark
import csv

cap = pyshark.FileCapture("../captures/tls_big.pcapng", display_filter="tls")

prev_time = None
flow_start = None

with open("../dataset/metadata_raw.csv","w",newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["length","inter_arrival","record_type","flow_duration"])

    for pkt in cap:
        try:
            ts = float(pkt.sniff_timestamp)
            length = int(pkt.length)

            if flow_start is None:
                flow_start = ts

            inter = ts - prev_time if prev_time else 0
            prev_time = ts

            record = pkt.tls.record_content_type

            flow = ts - flow_start

            writer.writerow([length, inter, record, flow])
        except:
            pass

print("metadata_raw.csv created")