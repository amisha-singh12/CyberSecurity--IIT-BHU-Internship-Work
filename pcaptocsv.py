import subprocess
import csv
import os
import re
import math
from collections import defaultdict, Counter
from statistics import mean

TSHARK_PATH = "C:\\Program Files\\Wireshark\\tshark.exe"

FIELDS = [
    "frame.time_epoch",
    "ip.src", "ip.dst",
    "udp.srcport", "udp.dstport", "tcp.srcport", "tcp.dstport",
    "ip.version", "ip.hdr_len", "ip.tos", "ip.len", "ip.ttl",
    "ip.proto", "ip.checksum", "udp.length", "udp.checksum",
    "icmp.type", "icmp.code", "icmp.checksum", "icmp.seq",
    "sip.Call-ID", "sip.Method", "sip.Status-Code",
    "sip.From", "sip.To", "sip.User-Agent", "sip.CSeq", 
    "sip.Via.branch", "sip.Max-Forwards",
    "rtcp", "rtp"
]

def run_tshark(pcap_path):
    tshark_cmd = [
        TSHARK_PATH, "-r", pcap_path, "-Y", "sip || rtp || rtcp || icmp", "-T", "fields"
    ]
    for field in FIELDS:
        tshark_cmd += ["-e", field]
    tshark_cmd += ["-E", "header=y", "-E", "separator=,", "-E", "quote=d", "-E", "occurrence=f"]
    result = subprocess.run(tshark_cmd, capture_output=True, text=True, check=True)
    return result.stdout.splitlines()

def parse_sip_sessions(tshark_output):
    sessions = defaultdict(list)
    headers = tshark_output[0].strip().split(',')
    for line in tshark_output[1:]:
        parts = line.strip().split(',')
        record = dict(zip(headers, parts))
        call_id = record.get("sip.Call-ID", "").strip('"')
        if call_id:
            sessions[call_id].append(record)
    return sessions

def entropy(strings):
    counter = Counter(strings)
    total = sum(counter.values())
    return -sum((count / total) * math.log2(count / total) for count in counter.values() if count > 0)

def extract_features_for_session(call_id, packets, attack_name):
    valid_methods = {
        "INVITE", "ACK", "BYE", "CANCEL", "OPTIONS", "REGISTER",
        "SUBSCRIBE", "NOTIFY", "INFO", "MESSAGE", "REFER", "UPDATE", "PRACK"
    }

    methods, statuses, agents, via_branches, cseqs, forwards = [], [], [], [], [], []
    invites, registers, responses_2xx, responses_4xx, acks = 0, 0, 0, 0, 0
    times, src_ips, dst_ips, from_uris, to_uris, ports = [], [], [], [], [], []
    rtp_present, rtcp_present = False, False
    packet_count = len(packets)

    for pkt in packets:
        ts = pkt.get("frame.time_epoch", "").strip('"')
        if ts:
            try:
                times.append(float(ts))
            except ValueError:
                continue

        method = pkt.get("sip.Method", "").strip('"').upper()
        status = pkt.get("sip.Status-Code", "").strip('"')
        ua = pkt.get("sip.User-Agent", "").strip('"')
        branch = pkt.get("sip.Via.branch", "").strip('"')
        cseq = pkt.get("sip.CSeq", "").strip('"')
        mf = pkt.get("sip.Max-Forwards", "").replace('"', '').strip()

        src_ips.append(pkt.get("ip.src", ""))
        dst_ips.append(pkt.get("ip.dst", ""))

        # Transport
        udp_src = pkt.get("udp.srcport", "")
        udp_dst = pkt.get("udp.dstport", "")
        tcp_src = pkt.get("tcp.srcport", "")
        tcp_dst = pkt.get("tcp.dstport", "")
        if udp_src and udp_dst:
            ports.append(udp_src + ":" + udp_dst)
        elif tcp_src and tcp_dst:
            ports.append(tcp_src + ":" + tcp_dst)

        from_uris.append(pkt.get("sip.From", ""))
        to_uris.append(pkt.get("sip.To", ""))

        if pkt.get("rtcp", ""):
            rtcp_present = True
        if pkt.get("rtp", ""):
            rtp_present = True

        if method in valid_methods:
            methods.append(method)

        if branch:
            via_branches.append(branch)

        if ua:
            agents.append(ua)

        if cseq:
            number = re.search(r'\d+', cseq)
            if number:
                cseqs.append(int(number.group()))

        if mf.isdigit():
            forwards.append(int(mf))

        if method == "INVITE": invites += 1
        elif method == "REGISTER": registers += 1
        elif method == "ACK": acks += 1

        if status.startswith("2"): responses_2xx += 1
        elif status.startswith("4"): responses_4xx += 1

    session_duration = max(times) - min(times) if times else 0
    avg_iat = mean([t2 - t1 for t1, t2 in zip(times[:-1], times[1:])] or [0])
    user_agent_entropy = entropy(agents)
    branch_reuse_rate = 1 - len(set(via_branches)) / len(via_branches) if via_branches else 0
    ack_missing_ratio = 1 - (acks / responses_2xx) if responses_2xx else 0
    cseq_gap = any(abs(c2 - c1) > 1000 for c1, c2 in zip(cseqs[:-1], cseqs[1:])) if len(cseqs) > 1 else False
    max_forwards_low = any(f < 5 for f in forwards)
    max_forwards_missing = len(forwards) < len(packets)
    suspicious_uri = any(re.search(r"sip:(admin|1000|test|root)@", uri, re.I) for uri in from_uris + to_uris)
    unusual_ports = any(p not in ["5060:5060", "5061:5061"] for p in ports)

    return {
        "call_id": call_id,
        "invite_count": invites,
        "register_count": registers,
        "response_2xx_count": responses_2xx,
        "response_4xx_count": responses_4xx,
        "sip_method_diversity": len(set(methods)),
        "sip_success_rate": round(responses_2xx / invites, 2) if invites else 0.0,
        "sip_session_duration": round(session_duration, 3),
        "avg_inter_arrival_time": round(avg_iat, 3),
        "user_agent_entropy": round(user_agent_entropy, 3),
        "branch_reuse_rate": round(branch_reuse_rate, 3),
        "ack_missing_ratio": round(ack_missing_ratio, 3),
        "rtcp_presence": int(rtcp_present),
        "unusual_port_flag": int(unusual_ports),
        "sip_uri_suspicion_flag": int(suspicious_uri),
        "cseq_gap_anomaly": int(cseq_gap),
        "max_forwards_low_flag": int(max_forwards_low),
        "max_forwards_missing_flag": int(max_forwards_missing),
        "src_ip": src_ips[0] if src_ips else "",
        "dst_ip": dst_ips[0] if dst_ips else "",
        "from_uri": from_uris[0] if from_uris else "",
        "to_uri": to_uris[0] if to_uris else "",
        "packet_count": packet_count,
        "class": attack_name
    }

def process_folder(root_folder, output_csv):
    header_written = False
    with open(output_csv, "w", newline="") as f:
        writer = None
        for dirpath, _, filenames in os.walk(root_folder):
            attack_name = os.path.basename(dirpath)
            for file in filenames:
                if not file.endswith(('.pcap', '.pcapng')):
                    continue
                path = os.path.join(dirpath, file)
                try:
                    tshark_output = run_tshark(path)
                    sessions = parse_sip_sessions(tshark_output)
                    for call_id, packets in sessions.items():
                        try:
                            row = extract_features_for_session(call_id, packets, attack_name)
                            if not header_written:
                                writer = csv.DictWriter(f, fieldnames=row.keys())
                                writer.writeheader()
                                header_written = True
                            writer.writerow(row)
                        except Exception as e:
                            print(f"⚠️ Skipped session {call_id} in {file} due to error: {e}")
                except Exception as e:
                    print(f"❌ Error processing {file} in {attack_name}: {e}")
    print(f"✅ Done. Features written to: {output_csv}")


if __name__ == "__main__":
    process_folder(
        "D:\\datasets\\featurecode\\pcapfiles1",
        "combined_sip_features1.csv"
    )
