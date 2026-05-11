interface Props {
  value: number;
  threshold?: number;
  size?: number;
}

export function ScoreGauge({ value, threshold = 0.85, size = 190 }: Props) {
  const cx = size / 2;
  const cy = size / 2 + 8;
  const r = size / 2 - 18;
  const SA = Math.PI;
  const EA = 2 * Math.PI;
  const t = Math.max(0, Math.min(1, value));
  const vA = SA + (EA - SA) * t;
  const thA = SA + (EA - SA) * threshold;
  const arc = (from: number, to: number, rad: number) => {
    const x1 = cx + Math.cos(from) * rad;
    const y1 = cy + Math.sin(from) * rad;
    const x2 = cx + Math.cos(to) * rad;
    const y2 = cy + Math.sin(to) * rad;
    return `M ${x1} ${y1} A ${rad} ${rad} 0 ${to - from > Math.PI ? 1 : 0} 1 ${x2} ${y2}`;
  };
  const passes = value >= threshold;
  const col = passes ? "var(--ivy)" : "var(--accent)";
  return (
    <svg width={size} height={size * 0.72} viewBox={`0 0 ${size} ${size * 0.72}`}>
      <path d={arc(SA, EA, r)} fill="none" stroke="var(--rule)" strokeWidth={1} />
      {Array.from({ length: 11 }).map((_, i) => {
        const a = SA + (EA - SA) * (i / 10);
        const x1 = cx + Math.cos(a) * r;
        const y1 = cy + Math.sin(a) * r;
        const x2 = cx + Math.cos(a) * (r - (i % 5 === 0 ? 8 : 4));
        const y2 = cy + Math.sin(a) * (r - (i % 5 === 0 ? 8 : 4));
        return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--ink-3)" strokeWidth={i % 5 === 0 ? 1.2 : 0.6} />;
      })}
      <path d={arc(SA, vA, r - 14)} fill="none" stroke={col} strokeWidth={6} strokeLinecap="butt" />
      <line
        x1={cx + Math.cos(thA) * (r - 22)}
        y1={cy + Math.sin(thA) * (r - 22)}
        x2={cx + Math.cos(thA) * (r + 4)}
        y2={cy + Math.sin(thA) * (r + 4)}
        stroke="var(--gold)"
        strokeWidth={2}
        strokeDasharray="3 2"
      />
      <line x1={cx} y1={cy} x2={cx + Math.cos(vA) * (r - 18)} y2={cy + Math.sin(vA) * (r - 18)} stroke="var(--ink)" strokeWidth={2} strokeLinecap="round" />
      <circle cx={cx} cy={cy} r={5} fill="var(--ink)" />
      <circle cx={cx} cy={cy} r={2} fill="var(--paper)" />
      <text x={cx} y={cy - 14} textAnchor="middle" fontFamily="Playfair Display,serif" fontSize="32" fontWeight="900" fill="var(--ink)">
        {(value * 100).toFixed(0)}
      </text>
      <text x={cx} y={cy - 30} textAnchor="middle" fontFamily="Inter,sans-serif" fontSize="8" fontWeight="700" letterSpacing=".14em" fill="var(--ink-3)">
        QUALITY · {(threshold * 100).toFixed(0)} REQ
      </text>
    </svg>
  );
}
