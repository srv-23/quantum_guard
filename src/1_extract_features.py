import pyshark
import pandas as pd
import numpy as np
import os
import sys

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCAP_DIR = os.path.join(BASE_DIR, "pcap")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

os.makedirs(DATASET_DIR, exist_ok=True)

def parse_pcap(pcap_path):
    """
    Parses a PCAP file using PyShark and yields raw packet dictionaries.
    Filters implicitly by only processing packets that have the required layers/fields.
    Focuses on IP and TCP layers, and TLS record type if available.
    """
    print(f"[INFO] Parsing {pcap_path}...")
    
    if not os.path.exists(pcap_path):
        print(f"[ERROR] File not found: {pcap_path}")
        return []

    # Filter for IP and TCP packets
    cap = pyshark.FileCapture(
        pcap_path,
        display_filter="tcp",
        keep_packets=False,
        include_raw=False
    )

    packets_list = []
    
    try:
        count = 0
        for pkt in cap:
            count += 1
            try:
                # Basic metadata extraction
                timestamp = float(pkt.sniff_timestamp)
                frame_len = int(pkt.length)
                
                # IP Layer
                if 'IP' in pkt:
                    src_ip = pkt.ip.src
                    dst_ip = pkt.ip.dst
                elif 'IPV6' in pkt:
                    src_ip = pkt.ipv6.src
                    dst_ip = pkt.ipv6.dst
                else:
                    continue 

                # TCP Layer
                if 'TCP' in pkt:
                    src_port = int(pkt.tcp.srcport)
                    dst_port = int(pkt.tcp.dstport)
                else:
                    continue 

                # TLS Layer (if present)
                # tls.record.content_type might be a list if multiple records in one packet
                tls_content_type = []
                if hasattr(pkt, 'tls'):
                    try:
                        # Use get_field_value if available or attribute access
                        if hasattr(pkt.tls, 'record_content_type'):
                             # It might be a single value or a collection if multiple records
                            val = pkt.tls.record_content_type
                            
                            # Convert to list of ints
                            if isinstance(val, str):
                                if ',' in val:
                                    tls_content_type = [int(x.strip()) for x in val.split(',')]
                                else:
                                    tls_content_type = [int(val)]
                            elif isinstance(val, (int, float)):
                                tls_content_type = [int(val)]
                            elif isinstance(val, list): # rare but possible in some parsers
                                tls_content_type = [int(x) for x in val]
                            else:
                                tls_content_type = [int(str(val))]

                    except Exception as e:
                        pass
                
                packet_dict = {
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'src_port': src_port,
                    'dst_port': dst_port,
                    'frame_len': frame_len,
                    'timestamp': timestamp,
                    'tls_content_type': tls_content_type
                }
                
                packets_list.append(packet_dict)

            except Exception as e:
                continue
                
    except Exception as e:
        print(f"[ERROR] reading pcap: {e}")
    finally:
        cap.close()
        
    print(f"[INFO] Parsed {len(packets_list)} packets from {os.path.basename(pcap_path)}")
    return packets_list

def extract_features(packet_list):
    """
    Aggregates packets into flows and computes statistical features.
    Flow key: (src_ip, dst_ip, src_port, dst_port)
    """
    if not packet_list:
        return pd.DataFrame()

    flows = {}
    
    print("[INFO] Aggregating packets into flows...")
    for pkt in packet_list:
        key = (pkt['src_ip'], pkt['dst_ip'], pkt['src_port'], pkt['dst_port'])
        
        if key not in flows:
            flows[key] = {
                'packets': [],
                'bytes': [],
                'timestamps': [],
                'tls_content_types': [] # List of lists
            }
        
        flows[key]['packets'].append(pkt)
        flows[key]['bytes'].append(pkt['frame_len'])
        flows[key]['timestamps'].append(pkt['timestamp'])
        flows[key]['tls_content_types'].append(pkt['tls_content_type'])
    
    analyzed_flows = []
    
    print(f"[INFO] Computing features for {len(flows)} flows...")
    for key, data in flows.items():
        count = len(data['packets'])
        total_bytes = np.sum(data['bytes'])
        pkts_len = np.array(data['bytes'])
        timestamps = np.array(data['timestamps'])
        
        mean_len = np.mean(pkts_len)
        std_len = np.std(pkts_len)
        
        sorted_times = np.sort(timestamps)
        duration = sorted_times[-1] - sorted_times[0]
        
        if count > 1:
            iat = np.diff(sorted_times)
            mean_iat = np.mean(iat)
            std_iat = np.std(iat)
        else:
            mean_iat = 0.0
            std_iat = 0.0
            
        # TLS Ratios
        # "handshake_ratio = handshake_packets / total_packets"
        n_handshake_pkts = 0
        n_app_pkts = 0
        n_alert_pkts = 0
        
        for ct_list in data['tls_content_types']:
            # ct_list is a list of content types in that packet (e.g. [22, 22] or [23] or [])
            if 22 in ct_list:
                n_handshake_pkts += 1
            if 23 in ct_list:
                n_app_pkts += 1
            if 21 in ct_list:
                n_alert_pkts += 1
        
        handshake_ratio = n_handshake_pkts / count if count > 0 else 0
        app_data_ratio = n_app_pkts / count if count > 0 else 0
        alert_ratio = n_alert_pkts / count if count > 0 else 0
            
        features = {
            'src_ip': key[0],
            'dst_ip': key[1],
            'src_port': key[2],
            'dst_port': key[3],
            'total_packets': count,
            'total_bytes': total_bytes,
            'mean_packet_length': mean_len,
            'std_packet_length': std_len,
            'flow_duration': duration,
            'mean_inter_arrival_time': mean_iat,
            'std_inter_arrival_time': std_iat,
            'handshake_ratio': handshake_ratio,
            'application_data_ratio': app_data_ratio,
            'alert_ratio': alert_ratio
        }
        
        analyzed_flows.append(features)
        
    return pd.DataFrame(analyzed_flows)

def build_dataset(pcap_filename, dataset_name):
    """
    Orchestrates parsing, extraction, labeling and saving.
    """
    pcap_path = os.path.join(PCAP_DIR, pcap_filename)
    csv_path = os.path.join(DATASET_DIR, dataset_name)
    
    print(f"\n[SECTION] Processing {pcap_filename} -> {dataset_name}")
    
    # 1. Parse
    packets = parse_pcap(pcap_path)
    
    # 2. Extract
    df = extract_features(packets)
    
    if df.empty:
        print(f"[WARN] No flows extracted from {pcap_filename}")
        return

    # 3. Label
    # Apply labeling logic
    if pcap_filename == "train_benign.pcap":
        df['label'] = 0
    elif pcap_filename == "test_mixed.pcap":
        # New Heuristic: Alert > 0 OR Handshake Ratio > 0.15
        conditions = (df['alert_ratio'] > 0.0) | (df['handshake_ratio'] > 0.15)
        df['label'] = np.where(conditions, 1, 0)
    
    # 4. Save
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Saved {len(df)} flows to {csv_path}")
    print(f"Class Distribution for {dataset_name}:")
    print(df['label'].value_counts())

def main():
    print("=== 1. Extract Features Pipeline ===")
    
    # Process Train Benign
    build_dataset("train_benign.pcap", "train_benign.csv")
    
    # Process Test Mixed
    build_dataset("test_mixed.pcap", "test_mixed.csv")
    
    print("\n=== Extraction Complete ===")

if __name__ == "__main__":
    main()
