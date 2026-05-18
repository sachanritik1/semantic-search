const STORAGE_KEY = "rag-document-session";
const MAX_RECENT = 10;

export interface RecentDocument {
	documentId: string;
	source?: string;
	ingestedAt: string;
}

export interface DocumentSession {
	activeDocumentId: string | null;
	recent: RecentDocument[];
}

const emptySession = (): DocumentSession => ({
	activeDocumentId: null,
	recent: [],
});

export function loadDocumentSession(): DocumentSession {
	if (typeof window === "undefined") {
		return emptySession();
	}

	try {
		const raw = window.localStorage.getItem(STORAGE_KEY);
		if (!raw) {
			return emptySession();
		}

		const parsed = JSON.parse(raw) as DocumentSession;
		if (!parsed || typeof parsed !== "object") {
			return emptySession();
		}

		return {
			activeDocumentId:
				typeof parsed.activeDocumentId === "string"
					? parsed.activeDocumentId
					: null,
			recent: Array.isArray(parsed.recent)
				? parsed.recent.filter(
						(entry): entry is RecentDocument =>
							typeof entry?.documentId === "string" &&
							typeof entry?.ingestedAt === "string",
					)
				: [],
		};
	} catch {
		return emptySession();
	}
}

function saveDocumentSession(session: DocumentSession): void {
	if (typeof window === "undefined") {
		return;
	}

	window.localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
}

export function setActiveDocument(documentId: string | null): DocumentSession {
	const session = loadDocumentSession();
	const next: DocumentSession = {
		...session,
		activeDocumentId: documentId,
	};
	saveDocumentSession(next);
	return next;
}

export function addIngestedDocument(input: {
	documentId: string;
	source?: string;
}): DocumentSession {
	const session = loadDocumentSession();
	const entry: RecentDocument = {
		documentId: input.documentId,
		source: input.source,
		ingestedAt: new Date().toISOString(),
	};

	const recent = [
		entry,
		...session.recent.filter((doc) => doc.documentId !== input.documentId),
	].slice(0, MAX_RECENT);

	const next: DocumentSession = {
		activeDocumentId: input.documentId,
		recent,
	};
	saveDocumentSession(next);
	return next;
}
