import type { CriticScores } from "@/types";

interface Props {
  scores: CriticScores;
  threshold?: number;
  size?: number;
}

export function RadarChart({ scores, threshold = 0.85, size = 220 }: Props) {
  const dims: (keyof CriticScores)[] = ["completeness", "evidence", "depth", "accuracy", "structure"];
  const labels = ["Completeness", "Evidence", "Depth", "Accuracy", "Structure"];
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 36;
  const ang = (i: number) => -Math.PI / 2 + (i * 2 * Math.PI) / dims.length;
  const pt = (i: number, v: number): [number, number] => [
    cx + Math.cos(ang(i)) * r * v,
    cy + Math.sin(ang(i)) * r * v,
  ];

  const dataPts = dims.map((d, i) => pt(i, scores[d] ?? 0));
  const thPts = dims.map((_, i) => pt(i, threshold));
  const dataPath = dataPts.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x},${y}`).join(" ") + " Z";
  const thPath = thPts.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x},${y}`).join(" ") + " Z";

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <defs>
        <pattern id="rhatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="4" stroke="var(--accent)" strokeWidth="1" />
        </pattern>
      </defs>
      {[0.25, 0.5, 0.75, 1].map((p, i) => (
        <polygon
          key={i}
          points={dims
            .map((_, j) => `${cx + Math.cos(ang(j)) * r * p},${cy + Math.sin(ang(j)) * r * p}`)
            .join(" ")}
          fill="none"
          stroke="var(--rule)"
          strokeWidth={p === 1 ? 1 : 0.5}
          strokeDasharray={p === 1 ? "0" : "2 2"}
        />
      ))}
      {dims.map((_, i) => (
        <line
          key={i}
          x1={cx}
          y1={cy}
          x2={cx + Math.cos(ang(i)) * r}
          y2={cy + Math.sin(ang(i)) * r}
          stroke="var(--rule)"
          strokeWidth={0.5}
        />
      ))}
      <path d={thPath} fill="none" stroke="var(--gold)" strokeWidth={1.5} strokeDasharray="4 3" />
      <path
        d={dataPath}
        fill="url(#rhatch)"
        fillOpacity=".65"
        stroke="var(--accent)"
        strokeWidth={1.8}
        strokeLinejoin="round"
      />
      {dataPts.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={3} fill="var(--paper)" stroke="var(--accent)" strokeWidth={1.5} />
      ))}
      {labels.map((label, i) => {
        const lr = r + 22;
        const x = cx + Math.cos(ang(i)) * lr;
        const y = cy + Math.sin(ang(i)) * lr;
        const anchor = Math.abs(Math.cos(ang(i))) < 0.3 ? "middle" : Math.cos(ang(i)) > 0 ? "start" : "end";
        return (
          <g key={i}>
            <text
              x={x}
              y={y - 2}
              textAnchor={anchor}
              fontFamily="Inter,sans-serif"
              fontSize="9"
              fontWeight="700"
              letterSpacing=".08em"
              fill="var(--ink-3)"
              style={{ textTransform: "uppercase" }}
            >
              {label}
            </text>
            <text x={x} y={y + 9} textAnchor={anchor} fontFamily="JetBrains Mono,monospace" fontSize="10" fill="var(--ink)" fontWeight="600">
              {((scores[dims[i]] ?? 0) * 100).toFixed(0)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
