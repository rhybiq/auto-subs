import { Crosshair, Flame, Radio, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { GameEvent, GameEventKind } from "@/types";

const KIND_ICON: Record<GameEventKind, typeof Crosshair> = {
  gunfire: Crosshair,
  explosion: Flame,
  electronic_beep: Radio,
};

function formatTimestamp(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

/**
 * Detected CS2 sound events as a wrapping row of chips, mirroring
 * `SpeakerChips`. Unlike speakers (relabeled via a settings popover, never
 * removed), these are expected to have false positives — the classifier was
 * never trained on game audio — so each chip supports both reclassifying the
 * kind and deleting it outright.
 */
export function GameEventChips({
  gameEvents,
  onGameEventsChange,
}: {
  gameEvents: GameEvent[];
  onGameEventsChange: (updated: GameEvent[]) => void;
}) {
  const { t } = useTranslation();

  const kindLabel = (kind: GameEventKind) => t(`output.gameEvents.kind.${kind}`);

  function handleKindChange(id: number, kind: GameEventKind) {
    onGameEventsChange(gameEvents.map((e) => (e.id === id ? { ...e, kind } : e)));
  }

  function handleDelete(id: number) {
    onGameEventsChange(gameEvents.filter((e) => e.id !== id));
  }

  return (
    <div className="flex flex-wrap gap-1.5">
      {gameEvents.map((event) => {
        const Icon = KIND_ICON[event.kind];
        return (
          <div
            key={event.id}
            className="flex max-w-full items-center gap-1 rounded-full border bg-background py-1 pl-2.5 pr-1 text-xs"
          >
            <Icon className="size-3 shrink-0 text-muted-foreground" />
            <Select
              value={event.kind}
              onValueChange={(value) => handleKindChange(event.id, value as GameEventKind)}
            >
              <SelectTrigger className="h-5 min-w-0 border-none bg-transparent px-1 text-xs shadow-none focus-visible:ring-0 [&_svg]:size-3">
                <SelectValue>{kindLabel(event.kind)}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="gunfire">{kindLabel("gunfire")}</SelectItem>
                <SelectItem value="explosion">{kindLabel("explosion")}</SelectItem>
                <SelectItem value="electronic_beep">{kindLabel("electronic_beep")}</SelectItem>
              </SelectContent>
            </Select>
            <span className="shrink-0 text-muted-foreground">{formatTimestamp(event.start)}</span>
            <button
              type="button"
              onClick={() => handleDelete(event.id)}
              className="ml-0.5 flex size-4 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              aria-label={t("output.gameEvents.remove", "Remove")}
              title={t("output.gameEvents.remove", "Remove")}
            >
              <X className="size-3" />
            </button>
          </div>
        );
      })}
    </div>
  );
}
