import { useMemo, type ReactNode } from "react";

interface InlineCtx {
  onCitation?: (n: number) => void;
}

function renderInline(text: string, { onCitation }: InlineCtx): ReactNode[] {
  const parts: ReactNode[] = [];
  let last = 0;
  const re = /(\*\*([^*]+)\*\*)|(`([^`]+)`)|(\[(\d+)\])|(\[([^\]]+)\]\(([^)]+)\))/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    if (m[2] !== undefined) parts.push(<strong key={m.index}>{m[2]}</strong>);
    else if (m[4] !== undefined)
      parts.push(
        <code
          key={m.index}
          className="ar-mono px-1"
          style={{
            fontSize: ".92em",
            background: "var(--paper-2)",
            border: "1px solid var(--rule)",
            borderRadius: 2,
          }}
        >
          {m[4]}
        </code>
      );
    else if (m[6] !== undefined) {
      const n = parseInt(m[6], 10);
      parts.push(
        <button key={m.index} className="cite-chip" onClick={() => onCitation?.(n)}>
          {n}
        </button>
      );
    } else if (m[8] !== undefined)
      parts.push(
        <a
          key={m.index}
          href={m[9]}
          target="_blank"
          rel="noreferrer"
          style={{
            color: "var(--accent)",
            textDecoration: "underline",
            textDecorationColor: "var(--rule)",
            textUnderlineOffset: 3,
            textDecorationThickness: 1,
          }}
        >
          {m[8]}
        </a>
      );
    last = re.lastIndex;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

type Block =
  | { type: "h1" | "h2" | "p"; content: string }
  | { type: "ul"; items: string[] };

interface Props {
  source: string;
  onCitation?: (n: number) => void;
  dropCap?: boolean;
}

export function MarkdownView({ source, onCitation, dropCap = false }: Props) {
  const blocks = useMemo<Block[]>(() => {
    const lines = source.split("\n");
    const out: Block[] = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (line.startsWith("# ")) {
        out.push({ type: "h1", content: line.slice(2) });
        i++;
      } else if (line.startsWith("## ")) {
        out.push({ type: "h2", content: line.slice(3) });
        i++;
      } else if (line.startsWith("- ")) {
        const items: string[] = [];
        while (i < lines.length && lines[i].startsWith("- ")) {
          items.push(lines[i].slice(2));
          i++;
        }
        out.push({ type: "ul", items });
      } else if (line.trim() === "") {
        i++;
      } else {
        const para: string[] = [];
        while (
          i < lines.length &&
          lines[i].trim() !== "" &&
          !lines[i].startsWith("#") &&
          !lines[i].startsWith("- ")
        ) {
          para.push(lines[i]);
          i++;
        }
        out.push({ type: "p", content: para.join(" ") });
      }
    }
    return out;
  }, [source]);

  return (
    <div className={`md ${dropCap ? "drop-cap" : ""}`}>
      {blocks.map((b, i) => {
        if (b.type === "h1") return <h1 key={i}>{renderInline(b.content, { onCitation })}</h1>;
        if (b.type === "h2") return <h2 key={i}>{renderInline(b.content, { onCitation })}</h2>;
        if (b.type === "p") return <p key={i}>{renderInline(b.content, { onCitation })}</p>;
        if (b.type === "ul")
          return (
            <ul key={i}>
              {b.items.map((it, j) => (
                <li key={j}>{renderInline(it, { onCitation })}</li>
              ))}
            </ul>
          );
        return null;
      })}
    </div>
  );
}
