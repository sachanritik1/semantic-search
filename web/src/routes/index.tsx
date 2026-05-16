import { Link, createFileRoute } from "@tanstack/react-router";
import { Badge } from "#/components/ui/badge.tsx";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "#/components/ui/card.tsx";
import { useHealthQuery } from "#/lib/api/hooks.ts";

export const Route = createFileRoute("/")({
	component: HomePage,
});

const featureCards = [
	{
		to: "/ingest" as const,
		title: "Ingest",
		description: "Upload a PDF to chunk, embed, and index in Qdrant and SQLite.",
	},
	{
		to: "/ask" as const,
		title: "Ask",
		description:
			"Run the full hybrid RAG pipeline: enhance, retrieve, rerank, and generate.",
	},
	{
		to: "/tools" as const,
		title: "Tools",
		description:
			"Try enhance, token counting, prompt templates, LLM test, and self-consistency.",
	},
];

function HomePage() {
	const health = useHealthQuery();

	const apiOk = health.isSuccess && health.data?.status === "ok";

	return (
		<div className="page-wrap py-10">
			<div className="mb-8 space-y-3">
				<p className="island-kicker m-0">Hybrid RAG</p>
				<h1 className="m-0 font-serif text-3xl font-semibold tracking-tight text-[var(--sea-ink)] sm:text-4xl">
					Semantic Search
				</h1>
				<p className="m-0 max-w-2xl text-[var(--sea-ink-soft)]">
					Dense + sparse retrieval over your documents, with an LLM for answers
					and debugging tools.
				</p>
				<div className="flex items-center gap-2">
					<span className="text-sm text-[var(--sea-ink-soft)]">API status</span>
					{health.isLoading ? (
						<Badge variant="secondary">Checking…</Badge>
					) : apiOk ? (
						<Badge variant="success">Reachable</Badge>
					) : (
						<Badge variant="destructive">
							{health.error instanceof Error
								? health.error.message
								: "Unreachable"}
						</Badge>
					)}
				</div>
			</div>

			<div className="grid gap-4 sm:grid-cols-3">
				{featureCards.map((card) => (
					<Link key={card.to} to={card.to} className="no-underline">
						<Card className="h-full transition hover:border-[var(--lagoon)]/40">
							<CardHeader>
								<CardTitle className="text-[var(--sea-ink)]">
									{card.title}
								</CardTitle>
								<CardDescription>{card.description}</CardDescription>
							</CardHeader>
							<CardContent>
								<span className="text-sm font-medium text-[var(--lagoon-deep)]">
									Open →
								</span>
							</CardContent>
						</Card>
					</Link>
				))}
			</div>
		</div>
	);
}
