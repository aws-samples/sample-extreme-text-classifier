/**
 * Error handling types and utilities
 */

// Error types
export type ErrorType = 'network' | 'validation' | 'processing' | 'filesystem' | 'auth' | 'unknown';

// Base error interface
export interface AppError {
  type: ErrorType;
  message: string;
  details?: string;
  code?: string;
  recoverable: boolean;
  suggestedAction?: string;
  timestamp: Date;
  context?: Record<string, any>;
}

// Specific error types
export interface NetworkError extends AppError {
  type: 'network';
  statusCode?: number;
  endpoint?: string;
}

export interface ValidationError extends AppError {
  type: 'validation';
  field?: string;
  validationRules?: string[];
}

export interface ProcessingError extends AppError {
  type: 'processing';
  operation?: string;
  stage?: string;
}

export interface FileSystemError extends AppError {
  type: 'filesystem';
  filePath?: string;
  operation?: 'read' | 'write' | 'delete' | 'create';
}

export interface AuthError extends AppError {
  type: 'auth';
  requiredPermission?: string;
}

// Error severity levels
export type ErrorSeverity = 'low' | 'medium' | 'high' | 'critical';

// Error with severity
export interface SeverityError extends AppError {
  severity: ErrorSeverity;
}

// Error handler function type
export type ErrorHandler = (error: AppError) => void;

// Error recovery action
export interface ErrorRecoveryAction {
  label: string;
  action: () => void | Promise<void>;
  primary?: boolean;
}

// Error display options
export interface ErrorDisplayOptions {
  showDetails: boolean;
  showRecoveryActions: boolean;
  autoHide: boolean;
  autoHideDelay?: number;
}

// Error boundary state
export interface ErrorBoundaryState {
  hasError: boolean;
  error?: AppError;
  errorInfo?: any;
}

// Validation result
export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
  warnings?: ValidationError[];
}

// Form validation
export interface FieldValidation {
  field: string;
  rules: ValidationRule[];
}

export interface ValidationRule {
  type: 'required' | 'minLength' | 'maxLength' | 'pattern' | 'custom';
  value?: any;
  message: string;
  validator?: (value: any) => boolean;
}