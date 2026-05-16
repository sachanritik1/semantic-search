import { createFormHook } from "@tanstack/react-form";
import {
	FileField,
	SelectField,
	TextAreaField,
	TextField,
} from "#/components/form/form-fields.tsx";
import { SubmitButton } from "#/components/form/submit-button.tsx";
import { fieldContext, formContext } from "#/hooks/form-context.ts";

export const { useAppForm, withForm } = createFormHook({
	fieldContext,
	formContext,
	fieldComponents: {
		TextAreaField,
		TextField,
		FileField,
		SelectField,
	},
	formComponents: {
		SubmitButton,
	},
});
