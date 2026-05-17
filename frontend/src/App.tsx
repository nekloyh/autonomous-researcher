import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  ArrowRight,
  Check,
  ChevronDown,
  ChevronUp,
  Clock,
  Code as CodeIcon,
  Copy,
  Database,
  Download,
  ExternalLink,
  Feather,
  FileSearch,
  Globe,
  ListTree,
  Loader2,
  MessageSquare,
  Network,
  PanelLeft,
  PanelLeftClose,
  Plus,
  RefreshCw,
  Send,
  Settings,
  Share2,
  ShieldCheck,
  BookOpen,
} from "lucide-react";

import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

import { MarkdownView } from "@/components/MarkdownView";
import { RadarChart } from "@/components/RadarChart";
import { ScoreGauge } from "@/components/ScoreGauge";

import { streamResearch } from "@/api/research";
import { faviconUrl, fmt, truncate } from "@/lib/utils";
import type {
  Citation,
  Critique,
  Finding,
  Phase,
  StageKey,
  StageState,
  SubTask,
  TaskStatus,
} from "@/types";

const PRESETS = [
  "So sánh chiến lược AI giữa VNG và FPT trong 2024",
  "Phân tích thị trường EV Việt Nam 6 tháng đầu 2025",
  "How does Anthropic's Constitutional AI compare to RLHF?",
  "Tác động của Bitcoin halving 2024 đến thị trường",
  "Compare Apple Vision Pro vs Meta Quest 3 ecosystem",
];

/* ============================================================================
 * Primitives
 * ========================================================================== */

const STATUS_LABEL: Record<TaskStatus, string> = {
  pending: "Awaiting",
  running: "Developing",
  done: "Filed",
  failed: "Spiked",
};

function StatusPill({ status }: { status: TaskStatus }) {
  const label = STATUS_LABEL[status] ?? status;
  const col =
    status === "running"
      ? "var(--accent)"
      : status === "done"
      ? "var(--ivy)"
      : status === "failed"
      ? "var(--accent)"
      : "var(--ink-4)";
  return (
    <span className="ar-sans ar-sc inline-flex items-center gap-1.5" style={{ color: col, fontSize: 10, fontWeight: 700 }}>
      <span
        className={status === "running" ? "ar-pulse" : ""}
        style={{
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: col,
          boxShadow: status === "running" ? `0 0 0 3px ${col}33` : "none",
          flexShrink: 0,
        }}
      />
      {label}
    </span>
  );
}

interface ToolMarkProps {
  icon: typeof Globe;
  name: string;
  count: number;
  active: boolean;
}

function ToolMark({ icon: Icon, name, count, active }: ToolMarkProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className="inline-flex items-center gap-1 px-1.5 py-0.5 ar-mono"
          style={{
            fontSize: 10,
            color: active ? "var(--accent)" : "var(--ink-3)",
            opacity: active ? 1 : 0.55,
            borderBottom: active ? "1px solid var(--accent)" : "1px solid transparent",
          }}
        >
          <Icon className="w-3 h-3" strokeWidth={1.6} />
          <span className="ar-tab">{count ?? 0}</span>
        </span>
      </TooltipTrigger>
      <TooltipContent className="ar-sans text-xs" style={{ background: "var(--ink)", color: "var(--paper)", border: "none" }}>
        {name} · {count} call{count === 1 ? "" : "s"}
      </TooltipContent>
    </Tooltip>
  );
}

function ConfidencePips({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const filled = Math.round(value * 5);
  return (
    <div className="flex items-center gap-1.5">
      <div className="flex items-center gap-[2px]">
        {Array.from({ length: 5 }).map((_, i) => (
          <span
            key={i}
            style={{
              width: 7,
              height: 11,
              background: i < filled ? "var(--accent)" : "transparent",
              border: `1px solid ${i < filled ? "var(--accent)" : "var(--rule)"}`,
            }}
          />
        ))}
      </div>
      <span className="ar-mono ar-tab" style={{ fontSize: 11, color: "var(--ink-3)" }}>
        {pct}%
      </span>
    </div>
  );
}

function FaviconStack({ urls, max = 5 }: { urls: string[]; max?: number }) {
  const shown = urls.slice(0, max);
  const extra = urls.length - shown.length;
  return (
    <div className="flex items-center">
      <div className="flex -space-x-1.5">
        {shown.map((u, i) => (
          <span
            key={i}
            className="inline-flex items-center justify-center w-5 h-5 rounded-full overflow-hidden"
            style={{ background: "var(--paper-2)", border: "1px solid var(--rule)", zIndex: shown.length - i }}
          >
            <img src={faviconUrl(u)} alt="" className="w-3.5 h-3.5" />
          </span>
        ))}
      </div>
      {extra > 0 && (
        <span className="ar-mono ar-tab ml-1.5" style={{ fontSize: 10, color: "var(--ink-3)" }}>
          +{extra}
        </span>
      )}
    </div>
  );
}

/* ============================================================================
 * Stage cards
 * ========================================================================== */

const STAGE_META: Record<StageKey, { label: string; name: string; icon: typeof ListTree; letter: string }> = {
  planner: { label: "Section A", name: "The Plan", icon: ListTree, letter: "A" },
  researchers: { label: "Section B", name: "Field Reports", icon: Network, letter: "B" },
  synthesizer: { label: "Section C", name: "The Composition", icon: BookOpen, letter: "C" },
  critic: { label: "Section D", name: "The Editorial", icon: ShieldCheck, letter: "D" },
};

interface StageHeaderProps {
  stage: StageKey;
  status: TaskStatus;
  elapsed: number | null;
  tokens: number | null;
  expanded: boolean;
  onToggle: () => void;
}

function StageHeader({ stage, status, elapsed, tokens, expanded, onToggle }: StageHeaderProps) {
  const meta = STAGE_META[stage];
  return (
    <button onClick={onToggle} className="w-full text-left px-5 py-4 flex items-start gap-4">
      <div
        className="ar-display neu-in-sm flex items-center justify-center flex-shrink-0"
        style={{ width: 44, height: 44, fontSize: 26, fontWeight: 900, color: "var(--accent)" }}
      >
        {meta.letter}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="ar-sans ar-sc" style={{ fontSize: 10, fontWeight: 700, color: "var(--ink-3)" }}>
            {meta.label}
          </span>
          <span style={{ color: "var(--rule)" }}>·</span>
          <StatusPill status={status} />
        </div>
        <h3 className="ar-headline" style={{ fontSize: "1.2rem", color: "var(--ink)" }}>
          {meta.name}
        </h3>
      </div>
      <div className="flex flex-col items-end gap-1 flex-shrink-0">
        {elapsed != null && (
          <span className="ar-mono ar-tab flex items-center gap-1" style={{ fontSize: 10, color: "var(--ink-3)" }}>
            <Clock className="w-3 h-3" strokeWidth={1.6} />
            {fmt(elapsed)}
          </span>
        )}
        {tokens != null && (
          <span className="ar-mono ar-tab" style={{ fontSize: 10, color: "var(--ink-3)" }}>
            {tokens.toLocaleString()} tok
          </span>
        )}
      </div>
      <div className="neu-xs flex items-center justify-center flex-shrink-0" style={{ width: 28, height: 28, marginTop: 2 }}>
        {expanded ? (
          <ChevronUp className="w-4 h-4" strokeWidth={1.6} style={{ color: "var(--ink-3)" }} />
        ) : (
          <ChevronDown className="w-4 h-4" strokeWidth={1.6} style={{ color: "var(--ink-3)" }} />
        )}
      </div>
    </button>
  );
}

function StageCard({
  stage,
  status,
  elapsed,
  tokens,
  expanded,
  onToggle,
  children,
}: StageHeaderProps & { children: ReactNode }) {
  return (
    <div className="neu rounded-md overflow-hidden ar-fade">
      <StageHeader stage={stage} status={status} elapsed={elapsed} tokens={tokens} expanded={expanded} onToggle={onToggle} />
      {expanded && (
        <div className="px-5 pb-5 pt-2" style={{ borderTop: "1px solid var(--rule-soft)" }}>
          {children}
        </div>
      )}
    </div>
  );
}

function PlannerBody({ tasks, taskStatus }: { tasks: SubTask[]; taskStatus: Record<string, TaskStatus> }) {
  return (
    <div className="pt-3 space-y-2">
      <div className="ar-sans ar-sc" style={{ fontSize: 10, color: "var(--ink-3)", marginBottom: 8 }}>
        Query decomposed into {tasks.length} sub-stories, dependencies as noted
      </div>
      {tasks.map((t, i) => {
        const s = taskStatus[t.id] ?? "pending";
        const col = s === "running" ? "var(--accent)" : s === "done" ? "var(--ivy)" : "var(--ink-4)";
        return (
          <Tooltip key={t.id}>
            <TooltipTrigger asChild>
              <div className="neu-in-sm flex items-center gap-3 px-3 py-2.5 cursor-help" style={{ borderLeft: `3px solid ${col}` }}>
                <span className="ar-mono ar-tab" style={{ fontSize: 10, color: "var(--ink-4)", width: 24 }}>
                  §{i + 1}
                </span>
                <span className="ar-serif flex-1" style={{ fontSize: 13, color: "var(--ink)" }}>
                  {t.question}
                </span>
                {t.dependencies.length > 0 && (
                  <span className="ar-mono ar-sc" style={{ fontSize: 9, color: "var(--ink-4)" }}>
                    ← {t.dependencies.join(", ")}
                  </span>
                )}
                <StatusPill status={s} />
              </div>
            </TooltipTrigger>
            <TooltipContent
              className="ar-serif text-xs max-w-xs"
              style={{ background: "var(--ink)", color: "var(--paper)", border: "none" }}
            >
              <div className="ar-sans ar-sc" style={{ fontSize: 9, color: "var(--ink-4)", marginBottom: 4 }}>
                Rationale
              </div>
              {t.rationale || "—"}
            </TooltipContent>
          </Tooltip>
        );
      })}
    </div>
  );
}

function ResearcherCard({
  task,
  finding,
  onExpand,
  onTrace,
}: {
  task: SubTask;
  finding?: Finding;
  onExpand: () => void;
  onTrace: () => void;
}) {
  if (!finding) {
    return (
      <article className="neu-in-sm p-4 flex flex-col gap-3 min-h-[160px]">
        <div className="ar-sans ar-sc ar-pulse" style={{ fontSize: 10, color: "var(--ink-3)" }}>
          Reporter dispatched · gathering...
        </div>
        <h4 className="ar-headline" style={{ fontSize: 15, color: "var(--ink)" }}>
          {task.question}
        </h4>
        <div className="flex-1 flex items-center justify-center gap-2 ar-sans" style={{ fontSize: 12, color: "var(--ink-4)" }}>
          <Loader2 className="w-3.5 h-3.5 ar-spin" strokeWidth={1.6} /> On the wire...
        </div>
      </article>
    );
  }
  const tools = finding.tools;
  const sourceCount = finding.source_count ?? finding.sources.length;
  return (
    <article className="neu-sm p-4 flex flex-col gap-3 ar-fade">
      <div className="flex items-start justify-between gap-2">
        <div className="ar-sans ar-sc" style={{ fontSize: 10, color: "var(--ink-3)" }}>
          Dispatch · {task.id.toUpperCase()}
        </div>
        <ConfidencePips value={finding.confidence} />
      </div>
      <h4 className="ar-headline" style={{ fontSize: 15, lineHeight: 1.25, color: "var(--ink)" }}>
        {task.question}
      </h4>
      <p
        className="ar-serif"
        style={{
          fontSize: 13,
          lineHeight: 1.55,
          color: "var(--ink-2)",
          display: "-webkit-box",
          WebkitLineClamp: 3,
          WebkitBoxOrient: "vertical",
          overflow: "hidden",
        }}
      >
        {finding.excerpt ??
          `${sourceCount} source${sourceCount === 1 ? "" : "s"} consulted. Detailed excerpt is not streamed by the SSE summary — open the report when ready.`}
      </p>
      <div className="r-soft pt-2.5 flex items-center justify-between gap-2 mt-auto">
        {finding.sources.length > 0 ? (
          <FaviconStack urls={finding.sources} />
        ) : (
          <span className="ar-mono ar-tab" style={{ fontSize: 10, color: "var(--ink-3)" }}>
            {sourceCount} source{sourceCount === 1 ? "" : "s"}
          </span>
        )}
        {tools && (
          <div className="flex items-center gap-0.5">
            <ToolMark icon={Globe} name="Web search" count={tools.web_search} active={tools.web_search > 0} />
            <ToolMark icon={FileSearch} name="Fetch URL" count={tools.fetch_url} active={tools.fetch_url > 0} />
            <ToolMark icon={Database} name="Vector search" count={tools.vector_search} active={tools.vector_search > 0} />
            <ToolMark icon={CodeIcon} name="Python exec" count={tools.python_exec} active={tools.python_exec > 0} />
          </div>
        )}
      </div>
      <div className="flex items-center gap-2 pt-1">
        <button
          onClick={onExpand}
          className="neu-btn ar-sans ar-sc px-3 py-1.5"
          style={{ fontSize: 10, fontWeight: 700, color: "var(--ink-2)" }}
        >
          Read full
        </button>
        <button
          onClick={onTrace}
          className="neu-btn ar-sans ar-sc px-3 py-1.5"
          style={{ fontSize: 10, fontWeight: 700, color: "var(--ink-2)" }}
        >
          ReAct trace
        </button>
      </div>
    </article>
  );
}

function ResearchersBody({
  tasks,
  taskStatus,
  findings,
  onExpand,
  onTrace,
}: {
  tasks: SubTask[];
  taskStatus: Record<string, TaskStatus>;
  findings: Record<string, Finding>;
  onExpand: (id: string) => void;
  onTrace: (id: string) => void;
}) {
  const reporting = tasks.filter((t) => taskStatus[t.id] === "running" || taskStatus[t.id] === "done");
  return (
    <div className="pt-3">
      <div className="ar-sans ar-sc mb-3" style={{ fontSize: 10, color: "var(--ink-3)" }}>
        Wire reports · {reporting.length}/{tasks.length} desks reporting
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {tasks.map((t) => {
          const s = taskStatus[t.id] ?? "pending";
          if (s === "pending") {
            return (
              <article key={t.id} className="neu-in-sm p-4 min-h-[160px]">
                <div className="ar-sans ar-sc" style={{ fontSize: 10, color: "var(--ink-4)" }}>
                  Awaiting wire
                </div>
                <h4 className="ar-headline mt-2" style={{ fontSize: 15, color: "var(--ink-3)" }}>
                  {t.question}
                </h4>
              </article>
            );
          }
          return (
            <ResearcherCard
              key={t.id}
              task={t}
              finding={findings[t.id]}
              onExpand={() => onExpand(t.id)}
              onTrace={() => onTrace(t.id)}
            />
          );
        })}
      </div>
    </div>
  );
}

function SynthesizerBody({
  status,
  words,
  citations,
  draftPreview,
  onTogglePreview,
  draftText,
}: {
  status: TaskStatus;
  words: number;
  citations: number;
  draftPreview: boolean;
  onTogglePreview: () => void;
  draftText: string;
}) {
  return (
    <div className="pt-3">
      <div className="ar-sans ar-sc mb-3" style={{ fontSize: 10, color: "var(--ink-3)" }}>
        Composing room · merging dispatches into copy
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="neu-in-sm p-4">
          <div className="ar-sans ar-sc" style={{ fontSize: 10, color: "var(--ink-3)", marginBottom: 4 }}>
            Words written
          </div>
          <div className="ar-display ar-tab" style={{ fontSize: 30, fontWeight: 900, color: "var(--ink)" }}>
            {words.toLocaleString()}
          </div>
        </div>
        <div className="neu-in-sm p-4">
          <div className="ar-sans ar-sc" style={{ fontSize: 10, color: "var(--ink-3)", marginBottom: 4 }}>
            Citations placed
          </div>
          <div className="ar-display ar-tab" style={{ fontSize: 30, fontWeight: 900, color: "var(--ink)" }}>
            {citations}
          </div>
        </div>
      </div>
      {status === "done" && draftText && (
        <div className="mt-3">
          <button
            onClick={onTogglePreview}
            className="neu-btn ar-sans ar-sc px-3 py-1.5"
            style={{ fontSize: 10, fontWeight: 700, color: "var(--ink-2)" }}
          >
            {draftPreview ? "Hide proof" : "Show galley proof"}
          </button>
          {draftPreview && (
            <div
              className="mt-3 neu-in p-4 overflow-auto ar-scroll ar-serif"
              style={{ maxHeight: 260, fontSize: 13, lineHeight: 1.65, color: "var(--ink-2)" }}
            >
              <div className="ar-sans ar-sc" style={{ fontSize: 9, color: "var(--ink-4)", marginBottom: 8 }}>
                Galley proof — excerpt
              </div>
              {draftText.split("\n\n")[2] || draftText.split("\n")[0]}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function CriticBody({
  critic,
  replan,
  iteration,
}: {
  critic: Critique;
  replan: boolean;
  iteration: number;
}) {
  const threshold = critic.threshold ?? 0.85;
  return (
    <div className="pt-3">
      <div className="ar-sans ar-sc mb-3 flex items-center justify-between" style={{ fontSize: 10, color: "var(--ink-3)" }}>
        <span>Editorial review · iteration {iteration}</span>
        {critic.is_complete ? (
          <span style={{ color: "var(--ivy)" }}>✓ Approved for press</span>
        ) : (
          <span style={{ color: "var(--accent)" }}>↻ Returned for revision</span>
        )}
      </div>
      {critic.scores && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-center">
          <div className="neu-in-sm p-3 flex items-center justify-center">
            <RadarChart scores={critic.scores} threshold={threshold} size={220} />
          </div>
          <div className="neu-in-sm p-3 flex items-center justify-center">
            <ScoreGauge value={critic.quality_score} threshold={threshold} size={200} />
          </div>
        </div>
      )}
      {!critic.scores && (
        <div className="neu-in-sm p-4 flex items-center justify-center">
          <ScoreGauge value={critic.quality_score} threshold={threshold} size={200} />
        </div>
      )}
      {replan && (
        <div className="neu-sm p-3 mt-4" style={{ borderLeft: "3px solid var(--accent)" }}>
          <div className="ar-sans ar-sc" style={{ fontSize: 10, color: "var(--accent)", fontWeight: 700, marginBottom: 4 }}>
            Stop the press · replan triggered
          </div>
          <div className="ar-serif" style={{ fontSize: 13, color: "var(--ink-2)" }}>
            Quality fell short of {(threshold * 100).toFixed(0)}. Spawning a follow-up reporter to chase missing leads.
          </div>
        </div>
      )}
      {critic.missing_info.length > 0 && (
        <div className="mt-4">
          <div className="ar-sans ar-sc mb-2" style={{ fontSize: 10, color: "var(--ink-3)" }}>
            Editor's flags
          </div>
          <ul className="space-y-1.5">
            {critic.missing_info.map((m, i) => (
              <li key={i} className="ar-serif flex gap-2" style={{ fontSize: 13, lineHeight: 1.55, color: "var(--ink-2)" }}>
                <span className="ar-display font-bold" style={{ color: "var(--accent)" }}>
                  §
                </span>
                {m}
              </li>
            ))}
          </ul>
        </div>
      )}
      {critic.suggestions.length > 0 && (
        <div className="mt-3 r-soft pt-3">
          <div className="ar-sans ar-sc mb-2" style={{ fontSize: 10, color: "var(--ink-3)" }}>
            Marginalia
          </div>
          <ul className="space-y-1">
            {critic.suggestions.map((s, i) => (
              <li key={i} className="ar-garamond italic" style={{ fontSize: 13, color: "var(--ink-3)" }}>
                — {s}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/* ============================================================================
 * Report viewer
 * ========================================================================== */

function ReportViewer({
  report,
  citations,
  onRerun,
  tasks,
}: {
  query: string;
  report: string;
  citations: Citation[];
  onRerun: () => void;
  tasks: SubTask[];
  findings: Record<string, Finding>;
}) {
  const [tab, setTab] = useState<"story" | "plan" | "sources" | "trace">("story");
  const [copied, setCopied] = useState(false);
  const [rerunOpen, setRerunOpen] = useState(false);

  const handleCitation = useCallback((n: number) => {
    setTab("sources");
    setTimeout(() => {
      const el = document.getElementById(`cite-${n}`);
      if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 100);
  }, []);

  const handleCopy = () => {
    navigator.clipboard?.writeText(report);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const handleDownload = () => {
    const b = new Blob([report], { type: "text/markdown" });
    const u = URL.createObjectURL(b);
    const a = document.createElement("a");
    a.href = u;
    a.download = "research-report.md";
    a.click();
    URL.revokeObjectURL(u);
  };

  const today = new Date().toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" });

  return (
    <div className="neu rounded-md overflow-hidden h-full flex flex-col">
      <div className="px-7 pt-5 pb-4 r-dbl-bot">
        <div className="flex items-center justify-between mb-3">
          <div className="ar-sans ar-sc flex items-center gap-2" style={{ fontSize: 10, color: "var(--ink-3)" }}>
            <span className="font-bold" style={{ color: "var(--accent)" }}>
              FRONT PAGE
            </span>
            <span style={{ color: "var(--rule)" }}>·</span>
            <span>{today}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <button onClick={handleCopy} className="neu-btn p-2" title="Copy">
              {copied ? (
                <Check className="w-3.5 h-3.5" strokeWidth={1.8} style={{ color: "var(--ivy)" }} />
              ) : (
                <Copy className="w-3.5 h-3.5" strokeWidth={1.6} style={{ color: "var(--ink-3)" }} />
              )}
            </button>
            <button onClick={handleDownload} className="neu-btn p-2" title="Download">
              <Download className="w-3.5 h-3.5" strokeWidth={1.6} style={{ color: "var(--ink-3)" }} />
            </button>
            <button className="neu-btn p-2" title="Share">
              <Share2 className="w-3.5 h-3.5" strokeWidth={1.6} style={{ color: "var(--ink-3)" }} />
            </button>
            <Dialog open={rerunOpen} onOpenChange={setRerunOpen}>
              <DialogTrigger asChild>
                <button className="neu-btn p-2" title="Re-run">
                  <RefreshCw className="w-3.5 h-3.5" strokeWidth={1.6} style={{ color: "var(--ink-3)" }} />
                </button>
              </DialogTrigger>
              <DialogContent className="ar-sheet ar-app">
                <DialogHeader>
                  <DialogTitle className="ar-headline text-xl" style={{ color: "var(--ink)" }}>
                    Run a fresh edition?
                  </DialogTitle>
                  <DialogDescription className="ar-garamond italic" style={{ color: "var(--ink-3)" }}>
                    The current report will be archived and the wire reopened.
                  </DialogDescription>
                </DialogHeader>
                <div className="flex justify-end gap-2 pt-3">
                  <button
                    onClick={() => setRerunOpen(false)}
                    className="neu-btn ar-sans ar-sc px-4 py-2"
                    style={{ fontSize: 10, fontWeight: 700 }}
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => {
                      setRerunOpen(false);
                      onRerun?.();
                    }}
                    className="neu-stamp ar-sans ar-sc px-4 py-2 rounded-sm"
                    style={{ fontSize: 10, fontWeight: 700 }}
                  >
                    Re-print
                  </button>
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </div>

        <Tabs value={tab} onValueChange={(v) => setTab(v as typeof tab)}>
          <TabsList className="bg-transparent p-0 h-auto gap-1.5">
            {(
              [
                ["story", "The Story"],
                ["plan", "The Outline"],
                ["sources", "Sources"],
                ["trace", "Process"],
              ] as const
            ).map(([v, l]) => (
              <TabsTrigger
                key={v}
                value={v}
                className="ar-tab-pill ar-sans ar-sc px-3 py-1.5 rounded-sm neu-xs"
                style={{ fontSize: 10, fontWeight: 700, color: "var(--ink-3)", background: "var(--paper)" }}
              >
                {l}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </div>

      <div className="flex-1 overflow-auto ar-scroll px-7 py-6">
        <Tabs value={tab}>
          <TabsContent value="story" className="mt-0">
            <MarkdownView source={report} onCitation={handleCitation} dropCap />
            <div className="r-double mt-8 pt-4 ar-sans ar-sc flex items-center justify-between" style={{ fontSize: 10, color: "var(--ink-3)" }}>
              <span>— end of report —</span>
              <span>autonomous-researcher</span>
            </div>
          </TabsContent>

          <TabsContent value="plan" className="mt-0">
            <h2 className="ar-headline mb-4" style={{ fontSize: "1.75rem", color: "var(--ink)" }}>
              The Editorial Outline
            </h2>
            <div className="ar-sans ar-sc mb-4" style={{ fontSize: 10, color: "var(--ink-3)" }}>
              How the desk broke this query into {tasks.length} sub-stories
            </div>
            <div className="space-y-3">
              {tasks.map((t, i) => (
                <div key={t.id} className="neu-in-sm p-4">
                  <div className="ar-sans ar-sc" style={{ fontSize: 10, color: "var(--accent)", marginBottom: 6 }}>
                    §{i + 1} · {t.id}
                  </div>
                  <div className="ar-headline" style={{ fontSize: 15, color: "var(--ink)", marginBottom: 6 }}>
                    {t.question}
                  </div>
                  {t.rationale && (
                    <div className="ar-garamond italic" style={{ fontSize: 13, color: "var(--ink-3)" }}>
                      {t.rationale}
                    </div>
                  )}
                  {t.dependencies.length > 0 && (
                    <div className="ar-mono ar-tab mt-2" style={{ fontSize: 10, color: "var(--ink-4)" }}>
                      depends on: {t.dependencies.join(", ")}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="sources" className="mt-0">
            <h2 className="ar-headline mb-4" style={{ fontSize: "1.75rem", color: "var(--ink)" }}>
              Bibliography
            </h2>
            <div className="ar-sans ar-sc mb-4" style={{ fontSize: 10, color: "var(--ink-3)" }}>
              Whitelisted citations — every claim is backed
            </div>
            <ol className="space-y-3">
              {citations.map((c) => (
                <li key={c.n} id={`cite-${c.n}`} className="neu-in-sm p-3 flex items-start gap-3">
                  <div
                    className="ar-display font-black flex-shrink-0 flex items-center justify-center"
                    style={{ width: 32, height: 32, fontSize: 18, color: "var(--accent)" }}
                  >
                    [{c.n}]
                  </div>
                  <img src={faviconUrl(c.url)} alt="" className="w-4 h-4 mt-1 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="ar-headline" style={{ fontSize: 15, lineHeight: 1.3, color: "var(--ink)" }}>
                      {c.title}
                    </div>
                    <div className="ar-mono truncate mt-0.5" style={{ fontSize: 11, color: "var(--ink-3)" }}>
                      {truncate(c.url, 80)}
                    </div>
                  </div>
                  <div className="flex items-center gap-1 flex-shrink-0">
                    <button onClick={() => navigator.clipboard?.writeText(c.url)} className="neu-btn p-1.5">
                      <Copy className="w-3 h-3" strokeWidth={1.6} style={{ color: "var(--ink-3)" }} />
                    </button>
                    <a
                      href={c.url}
                      target="_blank"
                      rel="noreferrer"
                      className="neu-btn p-1.5 flex items-center justify-center"
                    >
                      <ExternalLink className="w-3 h-3" strokeWidth={1.6} style={{ color: "var(--ink-3)" }} />
                    </a>
                  </div>
                </li>
              ))}
            </ol>
          </TabsContent>

          <TabsContent value="trace" className="mt-0">
            <h2 className="ar-headline mb-4" style={{ fontSize: "1.75rem", color: "var(--ink)" }}>
              Behind the Byline
            </h2>
            <div className="ar-garamond italic" style={{ fontSize: 13, color: "var(--ink-3)" }}>
              Full ReAct traces are recorded server-side (LangSmith). The current SSE stream surfaces only summary events; check the backend for per-tool detail.
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

/* ============================================================================
 * Sidebar
 * ========================================================================== */

function Sidebar({
  collapsed,
  onToggle,
  onSelectPreset,
}: {
  collapsed: boolean;
  onToggle: () => void;
  onSelectPreset: (p: string) => void;
}) {
  return (
    <aside
      className={`flex flex-col transition-all duration-300 ${collapsed ? "w-14" : "w-72"}`}
      style={{ borderRight: "1px solid var(--rule)" }}
    >
      <div className="p-3 flex items-center gap-2" style={{ borderBottom: "1px solid var(--rule-soft)" }}>
        <button onClick={onToggle} className="neu-btn p-2 flex-shrink-0">
          {collapsed ? (
            <PanelLeft className="w-4 h-4" strokeWidth={1.6} style={{ color: "var(--ink-3)" }} />
          ) : (
            <PanelLeftClose className="w-4 h-4" strokeWidth={1.6} style={{ color: "var(--ink-3)" }} />
          )}
        </button>
        {!collapsed && (
          <button
            className="flex-1 neu-stamp px-3 py-2 rounded-sm flex items-center gap-2 ar-sans ar-sc"
            style={{ fontSize: 10, fontWeight: 700 }}
          >
            <Plus className="w-3.5 h-3.5" strokeWidth={2} /> New edition
          </button>
        )}
      </div>
      {!collapsed && (
        <div className="flex-1 overflow-auto ar-scroll p-3 space-y-4">
          <div>
            <div className="ar-sans ar-sc px-1 mb-2" style={{ fontSize: 10, fontWeight: 700, color: "var(--ink-3)" }}>
              Suggested leads
            </div>
            <div className="space-y-1">
              {PRESETS.slice(0, 3).map((p, i) => (
                <button
                  key={i}
                  onClick={() => onSelectPreset(p)}
                  className="w-full text-left ar-garamond italic px-2 py-1.5"
                  style={{ fontSize: 12, color: "var(--ink-2)", borderBottom: "1px dotted var(--rule-soft)" }}
                >
                  {truncate(p, 40)}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}

/* ============================================================================
 * Idle view
 * ========================================================================== */

function IdleView({ onSubmit }: { onSubmit: (q: string) => void }) {
  const [query, setQuery] = useState("");
  const [advOpen, setAdvOpen] = useState(false);
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = ta.scrollHeight + "px";
  }, [query]);

  const submit = () => {
    if (query.trim()) onSubmit(query.trim());
  };
  const today = new Date().toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" });

  return (
    <div className="flex-1 overflow-auto ar-scroll">
      <div className="max-w-4xl mx-auto px-8 py-10">
        <div className="flex items-center justify-between ar-sans ar-sc" style={{ fontSize: 10, color: "var(--ink-3)" }}>
          <span>Vol. I · No. 47</span>
          <span>{today}</span>
          <span>research.ai · est. 2026</span>
        </div>

        <div className="r-double mt-2" />

        <div className="text-center pt-8 pb-2">
          <h1
            className="ar-headline"
            style={{ fontSize: "clamp(3rem,8vw,5.5rem)", fontWeight: 900, letterSpacing: "-.02em", lineHeight: 1.0, color: "var(--ink)" }}
          >
            The Autonomous
          </h1>
          <h1
            className="ar-headline"
            style={{
              fontSize: "clamp(3rem,8vw,5.5rem)",
              fontWeight: 900,
              letterSpacing: "-.02em",
              lineHeight: 1.0,
              color: "var(--accent)",
            }}
          >
            Researcher
          </h1>
          <div className="mt-3 ar-garamond italic" style={{ fontSize: "1.05rem", color: "var(--ink-3)" }}>
            "All the research that's fit to compile" — established 2026
          </div>
        </div>

        <div className="orn-rule mt-5 mb-8">
          <span className="ar-sans ar-sc" style={{ fontSize: 10, fontWeight: 700 }}>
            Today's Inquiry
          </span>
        </div>

        <div className="max-w-3xl mx-auto">
          <div className="neu-in rounded-md p-5">
            <div className="ar-sans ar-sc flex items-center gap-2 mb-3" style={{ fontSize: 10, color: "var(--ink-3)" }}>
              <Feather className="w-3 h-3" strokeWidth={1.8} /> Pose your inquiry · Vietnamese or English
            </div>
            <textarea
              ref={taRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                  e.preventDefault();
                  submit();
                }
              }}
              placeholder="What story shall we file today? Give me a question, a comparison, an investigation..."
              rows={3}
              className="w-full resize-none ar-serif bg-transparent outline-none"
              style={{ fontSize: "1.05rem", lineHeight: 1.65, color: "var(--ink)", minHeight: 80 }}
            />
            <div className="flex items-center justify-between r-soft mt-3 pt-3">
              <div className="flex items-center gap-2">
                <Dialog open={advOpen} onOpenChange={setAdvOpen}>
                  <DialogTrigger asChild>
                    <button className="neu-btn p-2">
                      <Settings className="w-3.5 h-3.5" strokeWidth={1.6} style={{ color: "var(--ink-3)" }} />
                    </button>
                  </DialogTrigger>
                  <DialogContent className="ar-sheet ar-app">
                    <DialogHeader>
                      <DialogTitle className="ar-headline text-xl" style={{ color: "var(--ink)" }}>
                        Edition settings
                      </DialogTitle>
                    </DialogHeader>
                    <div className="space-y-4 pt-2">
                      {(
                        [
                          ["Max iterations", "3"],
                          ["Editorial standard", "Strict — critic enabled (threshold 0.85)"],
                          ["Output language", "Auto · matches inquiry"],
                        ] as const
                      ).map(([k, v]) => (
                        <div key={k}>
                          <div className="ar-sans ar-sc mb-1.5" style={{ fontSize: 10, color: "var(--ink-3)" }}>
                            {k}
                          </div>
                          <div className="neu-in-sm px-3 py-2 ar-serif" style={{ fontSize: 13, color: "var(--ink)" }}>
                            {v}
                          </div>
                        </div>
                      ))}
                    </div>
                  </DialogContent>
                </Dialog>
              </div>
              <button
                onClick={submit}
                disabled={!query.trim()}
                className="neu-stamp ar-sans ar-sc px-5 py-2 rounded-sm flex items-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
                style={{ fontSize: 10, fontWeight: 700 }}
              >
                Send to press <ArrowRight className="w-3.5 h-3.5" strokeWidth={2} />
              </button>
            </div>
          </div>
          <div className="ar-mono ar-tab mt-2 text-right" style={{ fontSize: 10, color: "var(--ink-4)" }}>
            ⌘ + Enter to file
          </div>
        </div>

        <div className="mt-12">
          <div className="orn-rule mb-5">
            <span className="ar-sans ar-sc" style={{ fontSize: 10, fontWeight: 700 }}>
              Tips on the wire
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {PRESETS.map((p, i) => (
              <button
                key={i}
                onClick={() => setQuery(p)}
                className="text-left neu-sm p-4 hover:translate-y-[-1px] transition-transform"
              >
                <div className="ar-sans ar-sc" style={{ fontSize: 9, color: "var(--accent)", marginBottom: 6 }}>
                  Lead · {String(i + 1).padStart(2, "0")}
                </div>
                <div className="ar-headline" style={{ fontSize: 15, lineHeight: 1.3, color: "var(--ink)" }}>
                  {p}
                </div>
              </button>
            ))}
          </div>
        </div>

        <div className="mt-14">
          <div className="orn-rule mb-5">
            <span className="ar-sans ar-sc" style={{ fontSize: 10, fontWeight: 700 }}>
              How an edition is made
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {[
              { l: "A", n: "The Plan", d: "Decompose your inquiry into ≤5 sub-stories with dependencies." },
              { l: "B", n: "Field Reports", d: "Parallel reporters dispatched with web, fetch, vector, and exec tools." },
              { l: "C", n: "The Composition", d: "Editor weaves dispatches into a single, sourced story." },
              { l: "D", n: "The Editorial", d: "Critic scores 5 dimensions; replans if quality falls short." },
            ].map((s, i) => (
              <div key={i} className="neu-in-sm p-4">
                <div className="ar-display font-black" style={{ fontSize: 40, color: "var(--accent)", lineHeight: 1.0, marginBottom: 4 }}>
                  {s.l}
                </div>
                <div className="ar-headline" style={{ fontSize: 15, color: "var(--ink)", marginBottom: 6 }}>
                  {s.n}
                </div>
                <div className="ar-garamond italic" style={{ fontSize: 13, lineHeight: 1.5, color: "var(--ink-3)" }}>
                  {s.d}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div
          className="r-double mt-16 pt-3 ar-sans ar-sc flex items-center justify-between"
          style={{ fontSize: 10, color: "var(--ink-3)" }}
        >
          <span>The Autonomous Researcher · {new Date().getFullYear()}</span>
          <span>Set in Playfair Display & Source Serif 4</span>
        </div>
      </div>
    </div>
  );
}

/* ============================================================================
 * Helpers
 * ========================================================================== */

const INITIAL_STAGES: Record<StageKey, StageState> = {
  planner: { status: "pending", elapsed: null, tokens: null },
  researchers: { status: "pending", elapsed: null, tokens: null },
  synthesizer: { status: "pending", elapsed: null, tokens: null },
  critic: { status: "pending", elapsed: null, tokens: null },
};

function citationsFromUrls(urls: string[]): Citation[] {
  return urls.map((url, i) => {
    let domain = url;
    try {
      domain = new URL(url).hostname.replace(/^www\./, "");
    } catch {
      // ignore
    }
    return { n: i + 1, url, title: domain, domain };
  });
}

/* ============================================================================
 * Main
 * ========================================================================== */

export default function App() {
  const [evening, setEvening] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [phase, setPhase] = useState<Phase>("idle");
  const [query, setQuery] = useState("");
  const [stages, setStages] = useState<Record<StageKey, StageState>>(INITIAL_STAGES);
  const [tasks, setTasks] = useState<SubTask[]>([]);
  const [taskStatus, setTaskStatus] = useState<Record<string, TaskStatus>>({});
  const [findings, setFindings] = useState<Record<string, Finding>>({});
  const [iteration, setIteration] = useState(1);
  const [critic, setCritic] = useState<Critique | null>(null);
  const [replanFlag, setReplanFlag] = useState(false);
  const [synthDraft, setSynthDraft] = useState(false);
  const [synthWords, setSynthWords] = useState(0);
  const [synthCites, setSynthCites] = useState(0);
  const [finalReport, setFinalReport] = useState("");
  const [finalCitations, setFinalCitations] = useState<Citation[]>([]);
  const [expandedStages, setExpandedStages] = useState<Record<StageKey, boolean>>({
    planner: true,
    researchers: true,
    synthesizer: true,
    critic: true,
  });
  const [pipelineCollapsed, setPipelineCollapsed] = useState(false);
  const [traceTask, setTraceTask] = useState<string | null>(null);
  const [expandedTask, setExpandedTask] = useState<string | null>(null);
  const [followUpOpen, setFollowUpOpen] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const abortRef = useRef<(() => void) | null>(null);
  const startedAtRef = useRef<number>(0);

  const toggleStage = (k: StageKey) => setExpandedStages((p) => ({ ...p, [k]: !p[k] }));

  const togglePipelineCollapse = () => {
    const next = !pipelineCollapsed;
    setPipelineCollapsed(next);
    setExpandedStages({ planner: !next, researchers: !next, synthesizer: !next, critic: !next });
  };

  const reset = useCallback(() => {
    abortRef.current?.();
    abortRef.current = null;
    setPhase("idle");
    setQuery("");
    setStages(INITIAL_STAGES);
    setTasks([]);
    setTaskStatus({});
    setFindings({});
    setIteration(1);
    setCritic(null);
    setReplanFlag(false);
    setSynthDraft(false);
    setSynthWords(0);
    setSynthCites(0);
    setFinalReport("");
    setFinalCitations([]);
    setPipelineCollapsed(false);
    setExpandedStages({ planner: true, researchers: true, synthesizer: true, critic: true });
    setErrorMsg(null);
  }, []);

  const runLive = useCallback(
    async (q: string) => {
      reset();
      setQuery(q);
      setPhase("running");
      startedAtRef.current = Date.now();
      setStages((s) => ({ ...s, planner: { status: "running", elapsed: 0, tokens: null } }));

      const handle = streamResearch(q);
      abortRef.current = handle.abort;

      const elapsed = () => Date.now() - startedAtRef.current;

      try {
        for await (const ev of handle.events) {
          if (ev.event === "start") {
            // session_id available if needed
          } else if (ev.event === "update") {
            const u = ev.data;
            if (u.node === "planner" || u.node === "replan" || u.node === "gap_planner") {
              setIteration(u.iteration || 1);
              const nextTasks: SubTask[] =
                u.tasks && u.tasks.length > 0
                  ? u.tasks.map((t) => ({
                      id: t.id,
                      question: t.question,
                      rationale: t.rationale,
                      dependencies: t.dependencies ?? [],
                    }))
                  : Array.from({ length: u.plan_size }, (_, i) => ({
                      id: u.node === "gap_planner" ? `gap_${i + 1}` : `task_${i + 1}`,
                      question: u.node === "gap_planner" ? `Gap research ${i + 1}` : `Sub-task ${i + 1}`,
                      rationale: "",
                      dependencies: [],
                    }));
              setTasks(nextTasks);
              setTaskStatus(Object.fromEntries(nextTasks.map((p) => [p.id, "pending" as TaskStatus])));
              setStages((s) => ({
                ...s,
                planner: { status: "done", elapsed: elapsed(), tokens: u.tokens ?? null },
                researchers: { status: "running", elapsed: 0, tokens: null },
              }));
            } else if (u.node === "source_broker") {
              setStages((s) => ({
                ...s,
                planner: { status: "done", elapsed: elapsed(), tokens: null },
                researchers: { status: "running", elapsed: 0, tokens: null },
              }));
            } else if (u.node === "researcher") {
              const tid = u.task_id;
              setFindings((f) => ({
                ...f,
                [tid]: {
                  task_id: tid,
                  content: u.excerpt || "",
                  excerpt: u.excerpt,
                  confidence: u.confidence,
                  tool_calls: u.tool_calls ?? 0,
                  sources: [],
                  source_count: u.sources,
                },
              }));
              setTaskStatus((ts) => ({ ...ts, [tid]: "done" }));
            } else if (u.node === "synthesizer") {
              setSynthWords(u.draft_words);
              setSynthCites(u.citations);
              setStages((s) => ({
                ...s,
                researchers: { ...s.researchers, status: "done" },
                synthesizer: { status: "done", elapsed: elapsed(), tokens: u.tokens ?? null },
              }));
            } else if (u.node === "critic") {
              const c: Critique = {
                action: u.action,
                is_complete: u.is_complete,
                quality_score: u.score,
                missing_info: u.missing || [],
                gaps: u.gaps || [],
                suggestions: [],
                threshold: 0.85,
              };
              setCritic(c);
              setReplanFlag(!u.is_complete);
              setStages((s) => ({ ...s, critic: { status: "done", elapsed: elapsed(), tokens: u.tokens ?? null } }));
            } else if (u.node === "finalize") {
              // handled in done event
            }
          } else if (ev.event === "done") {
            setFinalReport(ev.data.final_report);
            setFinalCitations(citationsFromUrls(ev.data.citations));
            setPhase("done");
            setPipelineCollapsed(true);
            setExpandedStages({ planner: false, researchers: false, synthesizer: false, critic: false });
          } else if (ev.event === "error") {
            setErrorMsg(ev.data.error);
            setPhase("idle");
          }
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        if (msg !== "AbortError") {
          setErrorMsg(msg);
          setPhase("idle");
        }
      } finally {
        abortRef.current = null;
      }
    },
    [reset]
  );

  const handleSubmit = useCallback(
    (q: string) => {
      runLive(q);
    },
    [runLive]
  );

  const handleRerun = () => {
    runLive(query);
  };

  const totalElapsed = useMemo(
    () => Object.values(stages).reduce((a, s) => a + (s.elapsed ?? 0), 0),
    [stages]
  );
  const totalTokens = useMemo(
    () => Object.values(stages).reduce((a, s) => a + (s.tokens ?? 0), 0),
    [stages]
  );

  const displayCitations = finalCitations;
  const displayReport = finalReport;
  const displayTasks = tasks;

  return (
    <TooltipProvider delayDuration={150}>
      <div
        className={`ar-app ${evening ? "evening" : ""} h-screen flex flex-col overflow-hidden`}
        style={{ position: "relative", zIndex: 0 }}
      >
        <header
          className="flex items-center px-5 py-3 flex-shrink-0 relative"
          style={{ zIndex: 2, borderBottom: "3px double var(--ink)" }}
        >
          <div className="flex items-center gap-3">
            <div className="neu-xs flex items-center justify-center" style={{ width: 36, height: 36 }}>
              <Feather className="w-4 h-4" strokeWidth={1.5} style={{ color: "var(--accent)" }} />
            </div>
            <div className="leading-none">
              <div className="ar-headline" style={{ fontSize: "1.1rem", color: "var(--ink)" }}>
                The Autonomous Researcher
              </div>
              <div className="ar-sans ar-sc mt-0.5" style={{ fontSize: 9, color: "var(--ink-3)" }}>
                Vol. I · No. 47 · {new Date().toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
              </div>
            </div>
          </div>

          <div className="flex-1 flex justify-center">
            {phase !== "idle" && (
              <div className="neu-in-sm px-3 py-1.5 flex items-center gap-2.5">
                <StatusPill status={phase === "running" ? "running" : "done"} />
                <span style={{ color: "var(--rule)" }}>·</span>
                <span className="ar-mono ar-tab" style={{ fontSize: 10, color: "var(--ink-3)" }}>
                  {fmt(totalElapsed)}
                </span>
                {totalTokens > 0 && (
                  <>
                    <span style={{ color: "var(--rule)" }}>·</span>
                    <span className="ar-mono ar-tab" style={{ fontSize: 10, color: "var(--ink-3)" }}>
                      {totalTokens.toLocaleString()} tok
                    </span>
                  </>
                )}
                {iteration > 1 && (
                  <>
                    <span style={{ color: "var(--rule)" }}>·</span>
                    <span className="ar-sans ar-sc font-bold" style={{ fontSize: 10, color: "var(--accent)" }}>
                      Iter {iteration}
                    </span>
                  </>
                )}
              </div>
            )}
          </div>

          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setEvening((e) => !e)}
              className="neu-btn ar-sans ar-sc px-3 py-1.5 flex items-center gap-1.5"
              style={{ fontSize: 10, fontWeight: 700, color: "var(--ink-2)" }}
            >
              {evening ? "☾ Evening" : "☀ Morning"}
            </button>
            <button className="neu-btn p-2">
              <Settings className="w-3.5 h-3.5" strokeWidth={1.6} style={{ color: "var(--ink-3)" }} />
            </button>
          </div>
        </header>

        {errorMsg && (
          <div
            className="px-5 py-2 ar-sans ar-sc text-center"
            style={{
              fontSize: 11,
              fontWeight: 700,
              color: "#F5EBD9",
              background: "var(--accent)",
              position: "relative",
              zIndex: 3,
            }}
          >
            Wire failed: {errorMsg} — <button onClick={() => setErrorMsg(null)} className="underline">dismiss</button>
          </div>
        )}

        <div className="flex-1 flex overflow-hidden" style={{ position: "relative", zIndex: 2 }}>
          <Sidebar
            collapsed={sidebarCollapsed}
            onToggle={() => setSidebarCollapsed((c) => !c)}
            onSelectPreset={(p) => setQuery(p)}
          />

          <main className="flex-1 overflow-hidden flex">
            {phase === "idle" && <IdleView onSubmit={handleSubmit} />}

            {phase !== "idle" && (
              <div className="flex-1 flex overflow-hidden">
                <section
                  className={`flex flex-col overflow-hidden transition-all duration-300 ${
                    phase === "done" ? "w-2/5" : "w-1/2"
                  }`}
                  style={{ borderRight: "1px solid var(--rule)" }}
                >
                  <div className="px-5 py-4" style={{ borderBottom: "1px solid var(--rule-soft)" }}>
                    <div className="ar-sans ar-sc mb-1" style={{ fontSize: 10, color: "var(--ink-3)" }}>
                      Today's inquiry
                    </div>
                    <div className="ar-headline" style={{ fontSize: "1rem", lineHeight: 1.3, color: "var(--ink)" }}>
                      {query}
                    </div>
                    {phase === "done" && (
                      <button
                        onClick={togglePipelineCollapse}
                        className="mt-2 neu-btn ar-sans ar-sc px-2.5 py-1 flex items-center gap-1.5"
                        style={{ fontSize: 10, fontWeight: 700, color: "var(--ink-2)" }}
                      >
                        {pipelineCollapsed ? (
                          <ChevronDown className="w-3 h-3" strokeWidth={1.8} />
                        ) : (
                          <ChevronUp className="w-3 h-3" strokeWidth={1.8} />
                        )}
                        {pipelineCollapsed ? "Expand all" : "Collapse all"}
                      </button>
                    )}
                  </div>
                  <div className="flex-1 overflow-auto ar-scroll p-4 space-y-4">
                    <StageCard
                      stage="planner"
                      status={stages.planner.status}
                      elapsed={stages.planner.elapsed}
                      tokens={stages.planner.tokens}
                      expanded={expandedStages.planner}
                      onToggle={() => toggleStage("planner")}
                    >
                      {tasks.length > 0 ? (
                        <PlannerBody tasks={tasks} taskStatus={taskStatus} />
                      ) : (
                        <div className="ar-garamond italic py-3" style={{ fontSize: 13, color: "var(--ink-3)" }}>
                          The editor is reading your inquiry...
                        </div>
                      )}
                    </StageCard>
                    <StageCard
                      stage="researchers"
                      status={stages.researchers.status}
                      elapsed={stages.researchers.elapsed}
                      tokens={stages.researchers.tokens}
                      expanded={expandedStages.researchers}
                      onToggle={() => toggleStage("researchers")}
                    >
                      {tasks.length > 0 ? (
                        <ResearchersBody
                          tasks={tasks}
                          taskStatus={taskStatus}
                          findings={findings}
                          onExpand={setExpandedTask}
                          onTrace={setTraceTask}
                        />
                      ) : (
                        <div className="ar-garamond italic py-3" style={{ fontSize: 13, color: "var(--ink-3)" }}>
                          Waiting for assignments...
                        </div>
                      )}
                    </StageCard>
                    <StageCard
                      stage="synthesizer"
                      status={stages.synthesizer.status}
                      elapsed={stages.synthesizer.elapsed}
                      tokens={stages.synthesizer.tokens}
                      expanded={expandedStages.synthesizer}
                      onToggle={() => toggleStage("synthesizer")}
                    >
                      {stages.synthesizer.status === "running" || stages.synthesizer.status === "done" ? (
                        <SynthesizerBody
                          status={stages.synthesizer.status}
                          words={synthWords}
                          citations={synthCites}
                          draftPreview={synthDraft}
                          onTogglePreview={() => setSynthDraft((d) => !d)}
                          draftText={displayReport}
                        />
                      ) : (
                        <div className="ar-garamond italic py-3" style={{ fontSize: 13, color: "var(--ink-3)" }}>
                          The composing room is idle...
                        </div>
                      )}
                    </StageCard>
                    <StageCard
                      stage="critic"
                      status={stages.critic.status}
                      elapsed={stages.critic.elapsed}
                      tokens={stages.critic.tokens}
                      expanded={expandedStages.critic}
                      onToggle={() => toggleStage("critic")}
                    >
                      {critic ? (
                        <CriticBody critic={critic} replan={replanFlag} iteration={iteration} />
                      ) : (
                        <div className="ar-garamond italic py-3" style={{ fontSize: 13, color: "var(--ink-3)" }}>
                          The editor will review when copy arrives...
                        </div>
                      )}
                    </StageCard>
                  </div>
                </section>

                <section className="flex-1 overflow-hidden p-4">
                  {phase === "done" && displayReport ? (
                    <ReportViewer
                      query={query}
                      report={displayReport}
                      citations={displayCitations}
                      tasks={displayTasks}
                      findings={findings}
                      onRerun={handleRerun}
                    />
                  ) : (
                    <div className="h-full flex items-center justify-center">
                      <div className="text-center max-w-xs">
                        <div
                          className="ar-display font-black ar-pulse"
                          style={{ fontSize: "4.5rem", color: "var(--rule)", lineHeight: 1 }}
                        >
                          ※
                        </div>
                        <div className="ar-headline mt-3" style={{ fontSize: "1.1rem", color: "var(--ink)" }}>
                          The press is warm
                        </div>
                        <div className="ar-garamond italic mt-2" style={{ fontSize: 13, color: "var(--ink-3)" }}>
                          Your edition will appear here as soon as the editor signs off.
                        </div>
                      </div>
                    </div>
                  )}
                </section>
              </div>
            )}
          </main>
        </div>

        <Sheet open={!!expandedTask} onOpenChange={(o) => !o && setExpandedTask(null)}>
          <SheetContent
            side="right"
            className="ar-sheet ar-app w-full sm:max-w-xl"
            style={{ background: "var(--paper)", borderColor: "var(--rule)" }}
          >
            {expandedTask &&
              findings[expandedTask] &&
              (() => {
                const t = displayTasks.find((x) => x.id === expandedTask);
                const f = findings[expandedTask];
                if (!t) return null;
                return (
                  <>
                    <SheetHeader>
                      <div className="ar-sans ar-sc" style={{ fontSize: 10, color: "var(--accent)" }}>
                        Dispatch · {t.id.toUpperCase()}
                      </div>
                      <SheetTitle className="ar-headline text-2xl" style={{ color: "var(--ink)" }}>
                        {t.question}
                      </SheetTitle>
                      {t.rationale && (
                        <SheetDescription className="ar-garamond italic" style={{ color: "var(--ink-3)" }}>
                          {t.rationale}
                        </SheetDescription>
                      )}
                    </SheetHeader>
                    <div className="mt-5 space-y-4">
                      <div className="neu-in-sm p-3 flex items-center justify-between">
                        <span className="ar-sans ar-sc" style={{ fontSize: 10, color: "var(--ink-3)" }}>
                          Reliability
                        </span>
                        <ConfidencePips value={f.confidence} />
                      </div>
                      {f.excerpt && (
                        <div>
                          <div className="ar-sans ar-sc mb-2" style={{ fontSize: 10, color: "var(--ink-3)" }}>
                            The dispatch
                          </div>
                          <div className="ar-serif" style={{ fontSize: 14, lineHeight: 1.68, color: "var(--ink-2)" }}>
                            {f.excerpt}
                          </div>
                        </div>
                      )}
                      <div className="r-thin pt-3">
                        <div className="ar-sans ar-sc mb-2" style={{ fontSize: 10, color: "var(--ink-3)" }}>
                          Sources consulted
                        </div>
                        {f.sources.length > 0 ? (
                          <ul className="space-y-2">
                            {f.sources.map((s, i) => (
                              <li
                                key={i}
                                className="flex items-center gap-2 ar-mono"
                                style={{ fontSize: 11, color: "var(--ink-2)" }}
                              >
                                <img src={faviconUrl(s)} alt="" className="w-3.5 h-3.5" />
                                <a
                                  href={s}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="truncate"
                                  style={{
                                    color: "var(--accent)",
                                    textDecoration: "underline",
                                    textDecorationColor: "var(--rule)",
                                  }}
                                >
                                  {truncate(s, 60)}
                                </a>
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <div className="ar-serif" style={{ fontSize: 13, color: "var(--ink-3)" }}>
                            {f.source_count ?? 0} source{(f.source_count ?? 0) === 1 ? "" : "s"} consulted. Source URLs are available in the final report citations.
                          </div>
                        )}
                      </div>
                    </div>
                  </>
                );
              })()}
          </SheetContent>
        </Sheet>

        <Sheet open={!!traceTask} onOpenChange={(o) => !o && setTraceTask(null)}>
          <SheetContent
            side="right"
            className="ar-sheet ar-app w-full sm:max-w-2xl"
            style={{ background: "var(--paper)", borderColor: "var(--rule)" }}
          >
            <SheetHeader>
              <div className="ar-sans ar-sc" style={{ fontSize: 10, color: "var(--accent)" }}>
                Reporter's notebook · {(traceTask || "").toUpperCase()}
              </div>
              <SheetTitle className="ar-headline text-2xl" style={{ color: "var(--ink)" }}>
                ReAct trace
              </SheetTitle>
              <SheetDescription className="ar-garamond italic" style={{ color: "var(--ink-3)" }}>
                Per-step ReAct traces are written to LangSmith server-side, not surfaced over the SSE summary stream.
              </SheetDescription>
            </SheetHeader>
            <div className="mt-5 ar-garamond italic" style={{ fontSize: 13, color: "var(--ink-3)" }}>
              Open the LangSmith project linked in <span className="ar-mono">app/config.py</span> for the full thought/action/observation log.
            </div>
          </SheetContent>
        </Sheet>

        {phase === "done" && (
          <Sheet open={followUpOpen} onOpenChange={setFollowUpOpen}>
            <SheetTrigger asChild>
              <button
                className="fixed bottom-6 right-6 neu-stamp ar-sans ar-sc px-4 py-3 rounded-sm flex items-center gap-2"
                style={{ fontSize: 10, fontWeight: 700, zIndex: 20 }}
              >
                <MessageSquare className="w-3.5 h-3.5" strokeWidth={2} /> Letter to the editor
              </button>
            </SheetTrigger>
            <SheetContent
              side="right"
              className="ar-sheet ar-app w-full sm:max-w-md"
              style={{ background: "var(--paper)", borderColor: "var(--rule)" }}
            >
              <SheetHeader>
                <SheetTitle className="ar-headline text-2xl" style={{ color: "var(--ink)" }}>
                  Letter to the editor
                </SheetTitle>
                <SheetDescription className="ar-garamond italic" style={{ color: "var(--ink-3)" }}>
                  Press the desk for a follow-up — same wire, different angle.
                </SheetDescription>
              </SheetHeader>
              <div className="mt-5">
                <div className="neu-in rounded-md p-4">
                  <textarea
                    rows={5}
                    placeholder="What angle would you like us to chase next?"
                    className="w-full bg-transparent outline-none resize-none ar-serif"
                    style={{ fontSize: "1rem", color: "var(--ink)" }}
                  />
                </div>
                <button
                  className="mt-3 w-full neu-stamp py-2.5 rounded-sm ar-sans ar-sc flex items-center justify-center gap-2"
                  style={{ fontSize: 10, fontWeight: 700 }}
                >
                  <Send className="w-3.5 h-3.5" strokeWidth={2} /> File the follow-up
                </button>
              </div>
            </SheetContent>
          </Sheet>
        )}
      </div>
    </TooltipProvider>
  );
}
