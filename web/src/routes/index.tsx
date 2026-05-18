import { Link, createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { AskSection } from "#/components/rag/AskSection.tsx";
import { DocumentScopeBar } from "#/components/rag/DocumentScopeBar.tsx";
import { IngestSection } from "#/components/rag/IngestSection.tsx";
import { Badge } from "#/components/ui/badge.tsx";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "#/components/ui/card.tsx";
import { useHealthQuery } from "#/lib/api/hooks.ts";
import type { IngestResponse } from "#/lib/api/types.ts";
import {
	addIngestedDocument,
	loadDocumentSession,
	setActiveDocument,
	type DocumentSession,
} from "#/lib/session/documents.ts";

export const Route = createFileRoute("/")({
	component: HomePage,
});

function HomePage() {
	const health = useHealthQuery();
	const [session, setSession] = useState<DocumentSession>(() =>
		loadDocumentSession(),
	);
	useEffect(() => {
		setSession(loadDocumentSession());
	}, []);

	const apiOk = health.isSuccess && health.data?.status === "ok";

	const handleIngested = (result: IngestResponse, source?: string) => {
		setSession(
			addIngestedDocument({
				documentId: result.document_id,
				source,
			}),
		);
	};

	const handleActiveDocumentChange = (documentId: string | null) => {
		setSession(setActiveDocument(documentId));
	};

	return (
		<div className="page-wrap space-y-10 py-10">
			<div className="space-y-3">
				<p className="island-kicker m-0">Hybrid RAG</p>
				<h1 className="m-0 font-serif text-3xl font-semibold tracking-tight text-(--sea-ink) sm:text-4xl">
					Semantic Search
				</h1>
				<p className="m-0 max-w-2xl text-(--sea-ink-soft)">
					Ingest a PDF, then ask questions scoped to that document. Dense and
					sparse retrieval filter by the active document id.
				</p>
				<div className="flex items-center gap-2">
					<span className="text-sm text-(--sea-ink-soft)">API status</span>
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

			<DocumentScopeBar
				session={session}
				onActiveDocumentChange={handleActiveDocumentChange}
			/>

			<IngestSection onIngested={handleIngested} />

			<AskSection documentId={session.activeDocumentId} />

			<div className="grid gap-4 sm:grid-cols-1">
				<Link to="/tools" className="no-underline">
					<Card className="h-full transition hover:border-(--lagoon)/40">
						<CardHeader>
							<CardTitle className="text-(--sea-ink)">Tools</CardTitle>
							<CardDescription>
								Try enhance, token counting, prompt templates, LLM test, and
								self-consistency.
							</CardDescription>
						</CardHeader>
						<CardContent>
							<span className="text-sm font-medium text-(--lagoon-deep)">
								Open →
							</span>
						</CardContent>
					</Card>
				</Link>
			</div>
		</div>
	);
}
