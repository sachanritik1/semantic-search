import { useEffect, useRef, useState } from "react";
import { ApiErrorAlert } from "#/components/rag/ApiErrorAlert.tsx";
import { AnswerContent } from "#/components/rag/AnswerContent.tsx";
import { WorkspacePanel } from "#/components/rag/WorkspacePanel.tsx";
import { FieldGroup } from "#/components/ui/field.tsx";
import { useAppForm } from "#/hooks/use-app-form.ts";
import { askStream } from "#/lib/api/endpoints.ts";
import { ApiError } from "#/lib/api/client.ts";
import type { AskStreamMeta, AskStreamStage } from "#/lib/api/types.ts";
import { questionSchema } from "#/lib/forms/schemas.ts";

type AskStatus = "idle" | "preparing" | "streaming" | "done" | "error";

interface AskSectionProps {
	documentId: string | null;
}

const STAGE_ORDER: AskStreamStage[] = [
	"enhancing_query",
	"retrieving",
	"reranking",
	"generating",
];

const STAGE_LABELS: Record<AskStreamStage, string> = {
	enhancing_query: "Enhancing query",
	retrieving: "Retrieving documents",
	reranking: "Reranking results",
	generating: "Generating answer",
};

export function AskSection({ documentId }: AskSectionProps) {
	const [status, setStatus] = useState<AskStatus>("idle");
	const [streamingText, setStreamingText] = useState("");
	const [meta, setMeta] = useState<AskStreamMeta | null>(null);
	const [error, setError] = useState<Error | null>(null);
	const [retryAttempt, setRetryAttempt] = useState(0);
	const [stage, setStage] = useState<AskStreamStage | null>(null);
	const [completedStages, setCompletedStages] = useState<Set<AskStreamStage>>(
		() => new Set(),
	);
	const abortRef = useRef<AbortController | null>(null);
	const maxAttempts = 3;
	const canAsk = Boolean(documentId);

	useEffect(() => {
		return () => {
			abortRef.current?.abort();
		};
	}, []);

	const form = useAppForm({
		defaultValues: { question: "" },
		validators: { onSubmit: questionSchema },
		onSubmit: async ({ value }) => {
			if (!documentId) {
				return;
			}

			abortRef.current?.abort();
			const controller = new AbortController();
			abortRef.current = controller;

			setStatus("preparing");
			setStreamingText("");
			setMeta(null);
			setError(null);
			setRetryAttempt(0);
			setStage(null);
			setCompletedStages(new Set());

			try {
				await askStream(
					value.question,
					documentId,
					{
						onStatus: ({ stage: next }) => {
							setStage((current) => {
								if (current && current !== next) {
									setCompletedStages((prev) => {
										const updated = new Set(prev);
										updated.add(current);
										return updated;
									});
								}
								return next;
							});
						},
						onMeta: (payload) => {
							setMeta(payload);
							setStreamingText("");
							setRetryAttempt(0);
						},
						onRetry: (attempt) => {
							setRetryAttempt(attempt);
							setStatus("preparing");
							setStage(null);
							setCompletedStages(new Set());
						},
						onToken: (text) => {
							setStatus((current) =>
								current === "preparing" ? "streaming" : current,
							);
							setStreamingText((current) => current + text);
						},
						onDone: () => {
							setStage((current) => {
								if (current) {
									setCompletedStages((prev) => {
										const updated = new Set(prev);
										updated.add(current);
										return updated;
									});
								}
								return null;
							});
							setStatus("done");
						},
						onError: (message) => {
							setError(new Error(message));
							setStatus("error");
						},
					},
					controller.signal,
				);
				setStatus((current) => (current === "error" ? "error" : "done"));
			} catch (err) {
				if (controller.signal.aborted) {
					return;
				}
				setError(err instanceof Error ? err : new Error(String(err)));
				setStatus("error");
			}
		},
	});

	const isBusy = status === "preparing" || status === "streaming";
	const hasAnswer =
		streamingText.length > 0 || status === "done" || status === "streaming";

	return (
		<section className="space-y-3">
			<p className="island-kicker m-0">Query path</p>
			<WorkspacePanel
				title="Ask"
				description="Runs enhance → retrieve → rerank → generate. Answer streams as tokens arrive."
			>
				{!canAsk ? (
					<p className="m-0 text-sm text-(--sea-ink-soft)">
						Ingest a PDF and select it above before asking.
					</p>
				) : null}

				<form
					id="ask-form"
					className="flex flex-col gap-4"
					onSubmit={(event) => {
						event.preventDefault();
						void form.handleSubmit();
					}}
				>
					<form.AppForm>
						<div className="flex flex-col gap-3 sm:flex-row sm:items-end">
							<FieldGroup className="flex-1 gap-4">
								<form.AppField name="question">
									{(field) => (
										<field.TextAreaField
											label="Question"
											placeholder="What are the main findings in the document?"
											rows={3}
										/>
									)}
								</form.AppField>
							</FieldGroup>
							<form.SubmitButton
								label="Ask"
								disabled={!canAsk || isBusy}
								className="sm:self-end"
							/>
						</div>
					</form.AppForm>

					{status === "error" && error ? (
						error instanceof ApiError ? (
							<ApiErrorAlert error={error} />
						) : (
							<p className="m-0 text-sm text-destructive">{error.message}</p>
						)
					) : null}
				</form>

				<div className="border-t border-(--line) pt-4">
					<p className="island-kicker m-0 mb-3">Answer</p>

					{retryAttempt > 0 ? (
						<p className="m-0 text-sm text-(--sea-ink-soft)">
							Connection lost — retrying (attempt {retryAttempt + 1} of{" "}
							{maxAttempts})…
						</p>
					) : null}

					{(status === "preparing" || status === "streaming") &&
					retryAttempt === 0 &&
					(stage !== null || completedStages.size > 0) ? (
						<StageTracker
							currentStage={stage}
							completedStages={completedStages}
							done={status === "streaming" && stage === null}
						/>
					) : null}

					{meta ? (
						<details
							open
							className="mb-4 rounded-lg border border-(--line) bg-(--chip-bg)/50 px-3 py-2 text-sm text-(--sea-ink-soft)"
						>
							<summary className="cursor-pointer font-medium text-(--sea-ink)">
								Query enhancement
							</summary>
							<div className="mt-2 space-y-2">
								<p className="m-0">
									<span className="font-medium">Original:</span>{" "}
									{meta.original_question}
								</p>
								{meta.enhanced_questions &&
								meta.enhanced_questions.length > 0 ? (
									<div className="m-0">
										<p className="m-0 font-medium">Enhanced queries:</p>
										<ul className="mt-1 list-inside list-disc space-y-1">
											{meta.enhanced_questions.map((q) => (
												<li key={q}>{q}</li>
											))}
										</ul>
									</div>
								) : (
									<p className="m-0">
										<span className="font-medium">Enhanced:</span>{" "}
										{meta.enhanced_question}
									</p>
								)}
								{meta.cache_hit ? (
									<p className="m-0 text-xs text-(--sea-ink-soft)">
										Served from semantic cache
									</p>
								) : null}
							</div>
						</details>
					) : null}

					{hasAnswer ? (
						<AnswerContent content={streamingText} embedded />
					) : null}

					{status === "idle" ? (
						<p className="m-0 text-sm text-(--sea-ink-soft)">
							Submit a question to see the answer here.
						</p>
					) : null}
				</div>
			</WorkspacePanel>
		</section>
	);
}

interface StageTrackerProps {
	currentStage: AskStreamStage | null;
	completedStages: Set<AskStreamStage>;
	done: boolean;
}

function StageTracker({
	currentStage,
	completedStages,
	done,
}: StageTrackerProps) {
	return (
		<ol className="m-0 mb-4 flex flex-col gap-1.5 rounded-lg border border-(--line) bg-(--chip-bg)/40 px-3 py-2 text-sm">
			{STAGE_ORDER.map((s) => {
				const isCompleted = completedStages.has(s) || (done && s === "generating");
				const isActive = currentStage === s && !isCompleted;
				const isPending = !isCompleted && !isActive;
				return (
					<li
						key={s}
						className="flex items-center gap-2"
						aria-current={isActive ? "step" : undefined}
					>
						<StageIcon
							state={
								isCompleted ? "done" : isActive ? "active" : "pending"
							}
						/>
						<span
							className={
								isCompleted
									? "text-(--sea-ink)"
									: isActive
										? "font-medium text-(--sea-ink)"
										: "text-(--sea-ink-soft)"
							}
						>
							{STAGE_LABELS[s]}
							{isActive ? "…" : ""}
						</span>
						{isPending ? null : null}
					</li>
				);
			})}
		</ol>
	);
}

function StageIcon({ state }: { state: "pending" | "active" | "done" }) {
	if (state === "done") {
		return (
			<span
				aria-hidden
				className="inline-flex h-4 w-4 items-center justify-center rounded-full bg-(--sea-ink)/15 text-[10px] text-(--sea-ink)"
			>
				✓
			</span>
		);
	}
	if (state === "active") {
		return (
			<span
				aria-hidden
				className="inline-block h-3 w-3 animate-pulse rounded-full bg-(--sea-ink)"
			/>
		);
	}
	return (
		<span
			aria-hidden
			className="inline-block h-3 w-3 rounded-full border border-(--line)"
		/>
	);
}
