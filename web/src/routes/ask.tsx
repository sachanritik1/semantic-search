import { createFileRoute } from "@tanstack/react-router";
import { ApiErrorAlert } from "#/components/rag/ApiErrorAlert.tsx";
import { ResultPanel } from "#/components/rag/ResultPanel.tsx";
import { FieldGroup } from "#/components/ui/field.tsx";
import { useAppForm } from "#/hooks/use-app-form.ts";
import { questionSchema } from "#/lib/forms/schemas.ts";
import { useAskMutation } from "#/lib/api/hooks.ts";

export const Route = createFileRoute("/ask")({
  component: AskPage,
});

function AskPage() {
  const ask = useAskMutation();

  const form = useAppForm({
    defaultValues: { question: "" },
    validators: { onSubmit: questionSchema },
    onSubmit: async ({ value }) => {
      await ask.mutateAsync(value.question);
    },
  });

  return (
    <div className="page-wrap py-10">
      <div className="mb-6 space-y-2">
        <p className="island-kicker m-0">Query path</p>
        <h1 className="m-0 text-2xl font-semibold text-(--sea-ink)">
          Ask a question
        </h1>
        <p className="m-0 text-sm text-(--sea-ink-soft)">
          Runs enhance → dense + sparse retrieval → rerank → generate. This may
          take 10–30 seconds.
        </p>
      </div>

      <form
        id="ask-form"
        className="island-shell max-w-2xl space-y-4"
        onSubmit={(event) => {
          event.preventDefault();
          void form.handleSubmit();
        }}
      >
        <form.AppForm>
          <FieldGroup>
            <form.AppField name="question">
              {(field) => (
                <field.TextAreaField
                  label="Question"
                  placeholder="What are the main findings in the document?"
                />
              )}
            </form.AppField>
          </FieldGroup>
          <form.SubmitButton label="Ask" />
        </form.AppForm>

        {ask.isError ? <ApiErrorAlert error={ask.error} /> : null}

        {ask.isSuccess ? (
          <div className="space-y-4">
            <ResultPanel title="Answer">{ask.data.response}</ResultPanel>
            <details className="island-shell text-sm text-(--sea-ink-soft)">
              <summary className="cursor-pointer font-medium text-(--sea-ink)">
                Query enhancement
              </summary>
              <div className="mt-3 space-y-2">
                <p className="m-0">
                  <span className="font-medium">Original:</span>{" "}
                  {ask.data.original_question}
                </p>
                <p className="m-0">
                  <span className="font-medium">Enhanced:</span>{" "}
                  {ask.data.enhanced_question}
                </p>
              </div>
            </details>
          </div>
        ) : null}
      </form>
    </div>
  );
}
