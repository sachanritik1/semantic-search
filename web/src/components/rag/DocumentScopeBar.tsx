import { Badge } from "#/components/ui/badge.tsx";
import { Label } from "#/components/ui/label.tsx";
import { Switch } from "#/components/ui/switch.tsx";
import type { DocumentSession } from "#/lib/session/documents.ts";

interface DocumentScopeBarProps {
	session: DocumentSession;
	searchAll: boolean;
	onSearchAllChange: (searchAll: boolean) => void;
	onActiveDocumentChange: (documentId: string | null) => void;
}

function formatDocumentLabel(
	documentId: string,
	source?: string,
): string {
	if (source) {
		return `${source} (${documentId.slice(0, 8)}…)`;
	}
	return documentId;
}

export function DocumentScopeBar({
	session,
	searchAll,
	onSearchAllChange,
	onActiveDocumentChange,
}: DocumentScopeBarProps) {
	const { activeDocumentId, recent } = session;

	return (
		<div className="island-shell flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
			<div className="space-y-2">
				<p className="m-0 text-sm font-medium text-(--sea-ink)">Document scope</p>
				{searchAll ? (
					<p className="m-0 text-sm text-(--sea-ink-soft)">
						Searching all indexed documents.
					</p>
				) : activeDocumentId ? (
					<div className="flex flex-wrap items-center gap-2">
						<Badge variant="secondary" className="font-mono text-xs">
							{activeDocumentId}
						</Badge>
						<span className="text-sm text-(--sea-ink-soft)">
							Questions use only this document.
						</span>
					</div>
				) : (
					<p className="m-0 text-sm text-(--sea-ink-soft)">
						Ingest a PDF to select a document for scoped search.
					</p>
				)}
			</div>

			<div className="flex flex-col gap-3 sm:items-end">
				<div className="flex items-center gap-2">
					<Switch
						id="search-all"
						checked={searchAll}
						onCheckedChange={onSearchAllChange}
					/>
					<Label htmlFor="search-all" className="text-sm text-(--sea-ink-soft)">
						Search all documents
					</Label>
				</div>

				{recent.length > 0 ? (
					<label className="flex w-full flex-col gap-1 sm:w-72">
						<span className="text-xs font-medium text-(--sea-ink-soft)">
							Recent documents
						</span>
						<select
							className="rounded-md border border-(--line) bg-transparent px-3 py-2 text-sm text-(--sea-ink) outline-none focus-visible:border-(--lagoon)"
							value={searchAll ? "" : (activeDocumentId ?? "")}
							disabled={searchAll}
							onChange={(event) => {
								const value = event.target.value;
								onActiveDocumentChange(value || null);
							}}
						>
							<option value="">Select a document…</option>
							{recent.map((doc) => (
								<option key={doc.documentId} value={doc.documentId}>
									{formatDocumentLabel(doc.documentId, doc.source)}
								</option>
							))}
						</select>
					</label>
				) : null}
			</div>
		</div>
	);
}
