import json
from pathlib import Path

RECORDS_DIR = Path("firstpass_records")
DOMAINS = ["biology", "chemistry", "neuroscience", "physics", "earth_science"]
WORD_THRESHOLD = 20  

def audit():
    print(f"🔍 Auditing {RECORDS_DIR} for hollow records...\n")
    
    overall_total = 0
    overall_hollow = 0

    for domain in DOMAINS:
        d_path = RECORDS_DIR / domain
        if not d_path.exists():
            print(f"[-] Domain '{domain}' folder not found. Skipping.")
            continue

        files = list(d_path.glob("*.json"))
        if not files:
            print(f"[-] No JSON files found in {domain}.")
            continue

        domain_total = 0
        domain_hollow = []
        
        for f_path in sorted(files):
            if f_path.name == "failed.jsonl":
                continue
                
            domain_total += 1
            overall_total += 1
            
            try:
                with open(f_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Logic: Check the 'meat' of the paper (Results and Methods)
                pc = data.get("paper_content", {})
                results_text = pc.get("results", "")
                methods_text = pc.get("methods", "")
                
                res_words = len(str(results_text).split())
                met_words = len(str(methods_text).split())
                
                # Criteria for "Hollow": If both major body sections are nearly empty
                if res_words < WORD_THRESHOLD and met_words < WORD_THRESHOLD:
                    domain_hollow.append(f_path.stem)
                    overall_hollow += 1
                    
            except Exception as e:
                print(f"  [ERROR] Could not read {f_path.name}: {e}")

        # Domain Reporting
        h_count = len(domain_hollow)
        success_rate = ((domain_total - h_count) / domain_total) * 100 if domain_total > 0 else 0
        
        print(f"[{domain.upper()}]")
        print(f"  - Total Files Found: {domain_total}")
        print(f"  - Hollow Files     : {h_count}")
        print(f"  - Integrity Rate   : {success_rate:.1f}%")
        
        if domain_hollow:
            print(f"  - Hollow IDs: {', '.join(domain_hollow[:15])}{'...' if h_count > 15 else ''}")
        print("-" * 40)

    # Grand Total
    if overall_total > 0:
        o_success = ((overall_total - overall_hollow) / overall_total) * 100
        print(f"\nOVERALL SUMMARY")
        print(f"Total Processed : {overall_total}")
        print(f"Total Hollow    : {overall_hollow}")
        print(f"Final Success   : {o_success:.1f}%")
        
        if overall_hollow == 0:
            print("\nDATASET IS GOLD: No hollow files detected!")
        else:
            print(f"\nAction Required: {overall_hollow} files need healing.")

if __name__ == "__main__":
    audit()