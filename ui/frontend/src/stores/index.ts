/**
 * Store exports - Central export point for all Zustand stores
 */

export { useDatasetStore } from './dataset-store';
export { useClassificationStore } from './classification-store';
export { useUIStore } from './ui-store';
export { useAttributeStore } from './attribute-store';

// Re-export types for convenience
export type {
  Dataset,
  DatasetMetadata,
  ClassDefinition,
  CreateDatasetRequest,
  UpdateDatasetRequest
} from '../types/dataset';

export type {
  ClassificationResult,
  ClassificationConfig,
  ClassifyTextRequest,
  ClassifyPDFRequest,
  PDFExtractionResult
} from '../types/classification';

export type {
  LoadingState,
  Toast,
  ModalState,
  NavigationSection,
  NavigationItem,
  Theme,
  UIPreferences,
  FormState,
  TableState,
  FileUploadState,
  WizardState
} from '../types/ui';

export type {
  ClassAttribute,
  AttributeRule,
  AttributeCondition,
  AttributeConditionType,
  GenerateAttributesRequest,
  SaveAttributesRequest,
  AttributeValidationResult
} from '../types/attributes';