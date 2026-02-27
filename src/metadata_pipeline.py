import pyshark
import pandas as pd
import numpy as np
import os
import time

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCAP_DIR = os.path.join(BASE_DIR, "pcap")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_packet_features(pcap_path):
    """
    Reads a PCAP file and yields raw packet details 
    needed for flow aggregation.
    """
    if not os.path.exists(pcap_path):
        print(f"  [Warning] File not found: {pcap_path}")
        return []

    print(f"  Reading {os.path.basename(pcap_path)}...")
    
    # We use a display filter to ensure we act on TLS traffic over TCP
    # Adjust filter as needed (e.g., 'tcp port 8443' if specific)
    # Using 'tls' ensures we see TLS records, but we need TCP/IP headers too.
    cap = pyshark.FileCapture(
        pcap_path,
        display_filter="tls", 
        keep_packets=False
    )

    packets_data = []
    
    try:
        for pkt in cap:
            try:
                # 1. TIME
                timestamp = float(pkt.sniff_timestamp)
                
                # 2. IP
                if hasattr(pkt, 'ip'):
                    src_ip = pkt.ip.src
                    dst_ip = pkt.ip.dst
                elif hasattr(pkt, 'ipv6'):
                    src_ip = pkt.ipv6.src
                    dst_ip = pkt.ipv6.dst
                else:
                    continue 

                # 3. TCP
                if hasattr(pkt, 'tcp'):
                    src_port = int(pkt.tcp.srcport)
                    dst_port = int(pkt.tcp.dstport)
                else:
                    continue

                # 4. TLS Record Length
                # pyshark might show multiple records in one packet as a list
                # pkt.tls.record_length could be '23, 56' -> string
                tls_lengths = []
                if hasattr(pkt, 'tls') and hasattr(pkt.tls, 'record_length'):
                    field = pkt.tls.record_length
                    if isinstance(field, str):
                        # If multiple records, sum regular lengths? Or treat as one packet?
                        # Requirement says "tls.record.length".
                        # We will sum them to represent the TLS payload size in this packet.
                        # Note: This is an approximation if multiple records exist.
                        val_str = field.replace(' ', '') # clean potential spaces
                        # It might be comma separated if multiple layers
                        # But typically pyshark exposes multi-value fields as objects or strings
                        if ',' in val_str:
                             tls_lengths = [int(v) for v in val_str.split(',')]
                        else:
                             tls_lengths = [int(val_str)]
                    else:
                        # Sometimes it's already int/float if single
                        tls_lengths = [int(pkt.tls.record_length)]
                
                total_tls_len = sum(tls_lengths) if tls_lengths else 0
                
                # Define Flow Key (5-tuple)
                # We typically group by (src_ip, dst_ip, src_port, dst_port)
                # protocol is implicitly TCP here
                flow_key = (src_ip, dst_ip, src_port, dst_port)
                
                packets_data.append({
                    'key': flow_key,
                    'ts': timestamp,
                    'len': total_tls_len
                })

            except (AttributeError, ValueError):
                continue
            except Exception:
                continue
    except Exception as e:
        print(f"  [Error] Reading pcap: {e}")
    finally:
        cap.close()
        
    return packets_data

def aggregate_flows(packet_list, label):
    """
    Groups packets by flow key and computes statistical features.
    """
    flows = {}
    
    # 1. Grouping
    for p in packet_list:
        k = p['key']
        if k not in flows:
            flows[k] = []
        flows[k].append(p)
        
    print(f"  Aggregating {len(flows)} unique flows...")
    
    flow_features = []
    
    # 2. Computation
    for k, pkts in flows.items():
        # pkts is list of dicts {'ts', 'len'}
        
        # Sort by time
        pkts.sort(key=lambda x: x['ts'])
        
        timestamps = [x['ts'] for x in pkts]
        lengths = [x['len'] for x in pkts]
        
        # Basic Counts
        total_pkts = len(pkts)
        total_bytes = sum(lengths)
        
        # Length Stats
        length_arr = np.array(lengths)
        mean_len = np.mean(length_arr)
        std_len = np.std(length_arr)
        max_len = np.max(length_arr)
        min_len = np.min(length_arr)
        
        # Time Stats
        duration = timestamps[-1] - timestamps[0]
        if total_pkts > 1:
            iat = np.diff(timestamps)
            mean_iat = np.mean(iat)
            std_iat = np.std(iat)
        else:
            mean_iat = 0.0
            std_iat = 0.0
            
        # Metadata
        src_ip, dst_ip, src_port, dst_port = k
        
        flow_features.append({
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': src_port,
            'dst_port': dst_port,
            'total_packets': total_pkts,
            'total_bytes': total_bytes,
            'mean_packet_length': mean_len,
            'std_packet_length': std_len,
            'max_packet_length': max_len,
            'min_packet_length': min_len,
            'flow_duration': duration,
            'mean_inter_arrival_time': mean_iat,
            'std_inter_arrival_time': std_iat,
            'label': label
        })
        
    return flow_features

def process_files(file_list, output_filename):
    """
    Process a list of (filename, label) tuples and save to CSV.
    """
    all_flows = []
    
    for fname, label in file_list:
        full_path = os.path.join(PCAP_DIR, fname)
        raw_packets = extract_packet_features(full_path)
        if raw_packets:
            flows = aggregate_flows(raw_packets, label)
            all_flows.extend(flows)
        else:
            print(f"  [Info] No packets found or empty file: {fname}")
            
    if not all_flows:
        print(f"[Results] No flows generated for {output_filename}. Checking paths...")
        return
        
    df = pd.DataFrame(all_flows)
    
    # Clean NaN
    df.fillna(0, inplace=True)
    
    # Save
    out_path = os.path.join(OUTPUT_DIR, output_filename)
    df.to_csv(out_path, index=False)
    
    # Summary
    print(f"\n[Success] Created {output_filename}")
    print(f"  Total Flows: {len(df)}")
    print(f"  Class Distribution:\n{df['label'].value_counts()}")
    print("-" * 30)

def main():
    start_time = time.time()
    print("=== metadata_pipeline.py Started ===")
    print(f"Looking for PCAPs in: {PCAP_DIR}")
    
    # Define Inputs
    train_files = [
        ("train_benign.pcap", 0),
        ("train_anomaly.pcap", 1)
    ]
    
    test_files = [
        ("test_benign.pcap", 0),
        ("test_anomaly.pcap", 1)
    ]
    
    # Execute
    print("\n--- Processing Training Set ---")
    process_files(train_files, "final_train.csv")
    
    print("\n--- Processing Testing Set ---")
    process_files(test_files, "final_test.csv")
    
    print(f"\nTotal Execution Time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
