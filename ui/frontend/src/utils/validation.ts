/**
 * Validation utilities
 */

import { ValidationResult, ValidationError, ValidationRule, FieldValidation } from '../types/errors';
import { ErrorHandler } from './error-handler';

export class ValidationUtils {
  /**
   * Validates a single field against rules
   */
  static validateField(value: any, rules: ValidationRule[]): ValidationError[] {
    const errors: ValidationError[] = [];

    for (const rule of rules) {
      const isValid = this.applyRule(value, rule);
      if (!isValid) {
        errors.push(
          ErrorHandler.createValidationError(rule.message, undefined, [rule.type])
        );
      }
    }

    return errors;
  }

  /**
   * Validates multiple fields
   */
  static validateFields(data: Record<string, any>, validations: FieldValidation[]): ValidationResult {
    const errors: ValidationError[] = [];

    for (const validation of validations) {
      const fieldValue = data[validation.field];
      const fieldErrors = this.validateField(fieldValue, validation.rules);
      
      fieldErrors.forEach(error => {
        error.field = validation.field;
        errors.push(error);
      });
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }

  /**
   * Applies a single validation rule
   */
  private static applyRule(value: any, rule: ValidationRule): boolean {
    switch (rule.type) {
      case 'required':
        return this.isRequired(value);
      
      case 'minLength':
        return this.hasMinLength(value, rule.value);
      
      case 'maxLength':
        return this.hasMaxLength(value, rule.value);
      
      case 'pattern':
        return this.matchesPattern(value, rule.value);
      
      case 'custom':
        return rule.validator ? rule.validator(value) : true;
      
      default:
        return true;
    }
  }

  /**
   * Checks if value is required (not empty)
   */
  private static isRequired(value: any): boolean {
    if (value === null || value === undefined) return false;
    if (typeof value === 'string') return value.trim().length > 0;
    if (Array.isArray(value)) return value.length > 0;
    return true;
  }

  /**
   * Checks minimum length
   */
  private static hasMinLength(value: any, minLength: number): boolean {
    if (typeof value === 'string') return value.length >= minLength;
    if (Array.isArray(value)) return value.length >= minLength;
    return true;
  }

  /**
   * Checks maximum length
   */
  private static hasMaxLength(value: any, maxLength: number): boolean {
    if (typeof value === 'string') return value.length <= maxLength;
    if (Array.isArray(value)) return value.length <= maxLength;
    return true;
  }

  /**
   * Checks if value matches pattern
   */
  private static matchesPattern(value: any, pattern: RegExp | string): boolean {
    if (typeof value !== 'string') return false;
    const regex = typeof pattern === 'string' ? new RegExp(pattern) : pattern;
    return regex.test(value);
  }

  /**
   * Common validation rules
   */
  static rules = {
    required: (message = 'This field is required'): ValidationRule => ({
      type: 'required',
      message,
    }),

    minLength: (length: number, message?: string): ValidationRule => ({
      type: 'minLength',
      value: length,
      message: message || `Must be at least ${length} characters`,
    }),

    maxLength: (length: number, message?: string): ValidationRule => ({
      type: 'maxLength',
      value: length,
      message: message || `Must be no more than ${length} characters`,
    }),

    email: (message = 'Please enter a valid email address'): ValidationRule => ({
      type: 'pattern',
      value: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
      message,
    }),

    url: (message = 'Please enter a valid URL'): ValidationRule => ({
      type: 'pattern',
      value: /^https?:\/\/.+/,
      message,
    }),

    custom: (validator: (value: any) => boolean, message: string): ValidationRule => ({
      type: 'custom',
      validator,
      message,
    }),
  };

  /**
   * Dataset-specific validations
   */
  static datasetValidations: FieldValidation[] = [
    {
      field: 'name',
      rules: [
        ValidationUtils.rules.required(),
        ValidationUtils.rules.minLength(2),
        ValidationUtils.rules.maxLength(100),
      ],
    },
    {
      field: 'description',
      rules: [
        ValidationUtils.rules.required(),
        ValidationUtils.rules.minLength(10),
        ValidationUtils.rules.maxLength(500),
      ],
    },
  ];

  /**
   * Class definition validations
   */
  static classValidations: FieldValidation[] = [
    {
      field: 'name',
      rules: [
        ValidationUtils.rules.required(),
        ValidationUtils.rules.minLength(2),
        ValidationUtils.rules.maxLength(50),
      ],
    },
    {
      field: 'description',
      rules: [
        ValidationUtils.rules.required(),
        ValidationUtils.rules.minLength(10),
        ValidationUtils.rules.maxLength(300),
      ],
    },
  ];

  /**
   * File upload validations
   */
  static validateFile(file: File, options: {
    maxSize?: number;
    allowedTypes?: string[];
  } = {}): ValidationResult {
    const errors: ValidationError[] = [];
    const { maxSize = 10 * 1024 * 1024, allowedTypes = ['application/pdf'] } = options;

    // Check file size
    if (file.size > maxSize) {
      errors.push(
        ErrorHandler.createValidationError(
          `File size must be less than ${Math.round(maxSize / 1024 / 1024)}MB`
        )
      );
    }

    // Check file type
    if (!allowedTypes.includes(file.type)) {
      errors.push(
        ErrorHandler.createValidationError(
          `File type must be one of: ${allowedTypes.join(', ')}`
        )
      );
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }
}