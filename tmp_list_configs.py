import re, os
from pathlib import Path

rows = []
for root, dirs, files in os.walk('logs'):
    for f in files:
        if f == 'brief_logs.txt':
            p = os.path.join(root, f)
            with open(p, encoding='utf-8') as fh:
                lines = fh.readlines()
            if len(lines) < 2:
                continue
            name = lines[1].strip()
            m_k = re.search(r'K_(\d+)', name)
            m_g = re.search(r'fpi_gmax([0-9.]+)', name)
            k = int(m_k.group(1)) if m_k else None
            g = float(m_g.group(1)) if m_g else None
            rows.append((root, k, g))

for r in sorted(rows, key=lambda x: (x[2] if x[2] is not None else -1, x[1] if x[1] is not None else -1)):
    print(r)
