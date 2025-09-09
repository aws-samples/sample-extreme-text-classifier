/**
 * Error handling utilities
 */

import { AppError, ErrorType, NetworkError, ValidationError, ProcessingError, FileSystemError, AuthError } from '../types/errors';

export class ErrorHandler {
  /**
   * Creates a standardized error object
   */
  static createError(
    type: ErrorType,
    message: string,
    options: Partial<AppError> = {}
  ): AppError {
    return {
      type,
      message,
      details: options.details,
      code: options.code,
      recoverable: options.recoverable ?? true,
      suggestedAction: options.suggestedAction,
      timestamp: new Date(),
      context: options.context,
    };
  }

  /**
   * Creates a network error
   */
  static createNetworkError(
    message: string,
    statusCode?: number,
    endpoint?: string
  ): { error: NetworkError } {
    return {
      error: {
        type: 'network',
        message,
        statusCode,
        endpoint,
        recoverable: statusCode !== 401 && statusCode !== 403,
        suggestedAction: this.getNetworkErrorSuggestion(statusCode),
        timestamp: new Date(),
      }
    };
  }

  /**
   * Creates a validation error
   */
  static createValidationError(
    message: string,
    field?: string,
    validationRules?: string[]
  ): ValidationError {
    return {
      type: 'validation',
      message,
      field,
      validationRules,
      recoverable: true,
      suggestedAction: 'Please correct the highlighted fields and try again.',
      timestamp: new Date(),
    };
  }

  /**
   * Creates a processing error
   */
  static createProcessingError(
    message: string,
    operation?: string,
    stage?: string
  ): ProcessingError {
    return {
      type: 'processing',
      message,
      operation,
      stage,
      recoverable: true,
      suggestedAction: 'Please try again. If the problem persists, try with different input.',
      timestamp: new Date(),
    };
  }

  /**
   * Creates a filesystem error
   */
  static createFileSystemError(
    message: string,
    filePath?: string,
    operation?: 'read' | 'write' | 'delete' | 'create'
  ): FileSystemError {
    return {
      type: 'filesystem',
      message,
      filePath,
      operation,
      recoverable: operation !== 'delete',
      suggestedAction: 'Please check your permissions and available storage space.',
      timestamp: new Date(),
    };
  }

  /**
   * Creates an authentication error
   */
  static createAuthError(
    message: string,
    requiredPermission?: string
  ): AuthError {
    return {
      type: 'auth',
      message,
      requiredPermission,
      recoverable: false,
      suggestedAction: 'Please contact your administrator for access.',
      timestamp: new Date(),
    };
  }

  /**
   * Converts unknown errors to AppError
   */
  static fromUnknown(error: unknown): AppError {
    if (error instanceof Error) {
      return this.createError('unknown', error.message, {
        details: error.stack,
        recoverable: true,
        suggestedAction: 'Please try again or contact support if the problem persists.',
      });
    }

    if (typeof error === 'string') {
      return this.createError('unknown', error);
    }

    return this.createError('unknown', 'An unexpected error occurred', {
      details: JSON.stringify(error),
      recoverable: true,
    });
  }

  /**
   * Gets user-friendly error messages
   */
  static getUserFriendlyMessage(error: AppError): string {
    const messages: Record<ErrorType, string> = {
      network: 'Unable to connect to the service. Please check your connection and try again.',
      validation: 'Please check your input and correct any errors.',
      processing: 'Processing failed. Please try again with different input.',
      filesystem: 'Unable to save or load data. Please check your storage space.',
      auth: 'You do not have permission to perform this action.',
      unknown: 'An unexpected error occurred. Please try again.',
    };

    return error.message || messages[error.type] || messages.unknown;
  }

  /**
   * Determines if an error is recoverable
   */
  static isRecoverable(error: AppError): boolean {
    return error.recoverable;
  }

  /**
   * Gets suggested actions for network errors
   */
  private static getNetworkErrorSuggestion(statusCode?: number): string {
    if (!statusCode) return 'Please check your internet connection and try again.';

    const suggestions: Record<number, string> = {
      400: 'Please check your input and try again.',
      401: 'Please log in and try again.',
      403: 'You do not have permission to perform this action.',
      404: 'The requested resource was not found.',
      408: 'The request timed out. Please try again.',
      429: 'Too many requests. Please wait a moment and try again.',
      500: 'Server error. Please try again later.',
      502: 'Service temporarily unavailable. Please try again later.',
      503: 'Service temporarily unavailable. Please try again later.',
    };

    return suggestions[statusCode] || 'Please try again later.';
  }

  /**
   * Converts ApiError to AppError
   */
  static fromApiError(apiError: any): AppError {
    return {
      type: apiError.type || 'unknown',
      message: apiError.message || 'An error occurred',
      details: apiError.details,
      code: apiError.code,
      recoverable: apiError.recoverable ?? true,
      suggestedAction: apiError.suggestedAction,
      timestamp: new Date(),
      context: apiError.context,
    };
  }

  /**
   * Logs error for debugging (in development)
   */
  static logError(error: AppError | any): void {
    // Convert ApiError to AppError if needed
    const appError = error.timestamp ? error : this.fromApiError(error);
    
    if (process.env.NODE_ENV === 'development') {
      console.group(`🚨 ${appError.type.toUpperCase()} ERROR`);
      console.error('Message:', appError.message);
      if (appError.details) console.error('Details:', appError.details);
      if (appError.context) console.error('Context:', appError.context);
      console.error('Recoverable:', appError.recoverable);
      if (appError.suggestedAction) console.info('Suggestion:', appError.suggestedAction);
      console.groupEnd();
    }
  }
}