import React, { useState, useCallback, useRef } from 'react'
import { Upload, File, X, CheckCircle, AlertCircle } from 'lucide-react'
import { Button } from './ui/Button'
import { Card, CardContent } from './ui/Card'
import { formatFileSize, validateFileType, validateFileSize } from '../lib/utils'

const UploadPDF = ({ onUpload, isLoading = false }) => {
  const [dragActive, setDragActive] = useState(false)
  const [selectedFiles, setSelectedFiles] = useState([])
  const inputRef = useRef(null)

  const allowedTypes = ['pdf']
  const maxSizeMB = 50

  const validateFile = useCallback((file) => {
    const errors = []
    
    if (!validateFileType(file, allowedTypes)) {
      errors.push(`File type not allowed. Only PDF files are supported.`)
    }
    
    if (!validateFileSize(file, maxSizeMB)) {
      errors.push(`File size exceeds ${maxSizeMB}MB limit.`)
    }
    
    return {
      valid: errors.length === 0,
      errors
    }
  }, [])

  const handleDrag = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    const files = Array.from(e.dataTransfer.files)
    handleFiles(files)
  }, [])

  const handleFiles = useCallback((files) => {
    const newFiles = files.map(file => {
      const validation = validateFile(file)
      return {
        id: Math.random().toString(36).substr(2, 9),
        file,
        name: file.name,
        size: file.size,
        valid: validation.valid,
        errors: validation.errors,
        status: validation.valid ? 'ready' : 'error'
      }
    })
    
    setSelectedFiles(prev => [...prev, ...newFiles])
  }, [validateFile])

  const handleFileInput = useCallback((e) => {
    const files = Array.from(e.target.files)
    handleFiles(files)
    e.target.value = '' // Reset input
  }, [handleFiles])

  const handleChooseFiles = useCallback(() => {
    inputRef.current?.click()
  }, [])

  const removeFile = useCallback((fileId) => {
    setSelectedFiles(prev => prev.filter(f => f.id !== fileId))
  }, [])

  const handleUpload = useCallback(async () => {
    const validFiles = selectedFiles.filter(f => f.valid)
    
    if (validFiles.length === 0) return

    setSelectedFiles(prev => prev.map(f => 
      f.valid ? { ...f, status: 'uploading' } : f
    ))

    try {
      await onUpload(validFiles.map(f => f.file))
      setSelectedFiles(prev => prev.map(f => 
        f.status === 'uploading' ? { ...f, status: 'uploaded' } : f
      ))
    } catch (error) {
      setSelectedFiles(prev => prev.map(f => 
        f.status === 'uploading' ? { ...f, status: 'error', errors: [error.message] } : f
      ))
    }
  }, [selectedFiles, onUpload])

  const clearAll = useCallback(() => {
    setSelectedFiles([])
  }, [])

  const validFiles = selectedFiles.filter(f => f.valid)
  const hasValidFiles = validFiles.length > 0
  const hasInvalidFiles = selectedFiles.some(f => !f.valid)

  return (
    <Card className="w-full">
      <CardContent className="p-6">
        <div className="space-y-6">
          {/* Upload Area */}
          <div
            className={`upload-area ${dragActive ? 'dragover' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <div className="flex flex-col items-center space-y-4">
              <div className="p-4 bg-blue-100 rounded-full">
                <Upload className="h-8 w-8 text-blue-600" />
              </div>
              
              <div className="text-center">
                <h3 className="text-lg font-semibold text-gray-900">
                  Upload Reference PDFs
                </h3>
                <p className="text-sm text-gray-500 mt-1">
                  Drag and drop your PDF files here, or click to browse
                </p>
                <p className="text-xs text-gray-400 mt-2">
                  Maximum file size: {maxSizeMB}MB per file
                </p>
              </div>

              {/* Hidden Input */}
              <input
                ref={inputRef}
                type="file"
                multiple
                accept=".pdf"
                onChange={handleFileInput}
                style={{ display: 'none' }}
              />

              <Button variant="outline" onClick={handleChooseFiles}>
                Choose Files
              </Button>
            </div>
          </div>

          {/* File List */}
          {selectedFiles.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium text-gray-900">
                  Selected Files ({selectedFiles.length})
                </h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={clearAll}
                  className="text-gray-500 hover:text-gray-700"
                >
                  Clear All
                </Button>
              </div>
              
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {selectedFiles.map((file) => (
                  <div key={file.id} className="file-list-item flex items-center justify-between">
                    <div className="flex items-center space-x-3 flex-1">
                      <File className="h-5 w-5 text-gray-400" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 truncate">
                          {file.name}
                        </p>
                        <p className="text-xs text-gray-500">
                          {formatFileSize(file.size)}
                        </p>
                        {file.errors && file.errors.length > 0 && (
                          <div className="mt-1">
                            {file.errors.map((error, index) => (
                              <p key={index} className="text-xs text-red-600">
                                {error}
                              </p>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="flex items-center space-x-2">
                      {file.status === 'ready' && <CheckCircle className="h-5 w-5 text-green-500" />}
                      {file.status === 'uploading' && <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>}
                      {file.status === 'uploaded' && <CheckCircle className="h-5 w-5 text-green-500" />}
                      {file.status === 'error' && <AlertCircle className="h-5 w-5 text-red-500" />}

                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeFile(file.id)}
                        className="h-8 w-8 p-0 text-gray-400 hover:text-red-600"
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Upload Button */}
          {hasValidFiles && (
            <div className="flex items-center justify-between pt-4 border-t">
              <div className="text-sm text-gray-600">
                {validFiles.length} file(s) ready to upload
                {hasInvalidFiles && (
                  <span className="text-red-600 ml-2">
                    ({selectedFiles.length - validFiles.length} invalid)
                  </span>
                )}
              </div>
              
              <Button
                onClick={handleUpload}
                disabled={isLoading || !hasValidFiles}
                className="min-w-32"
              >
                {isLoading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Uploading...
                  </>
                ) : (
                  `Upload ${validFiles.length} File${validFiles.length > 1 ? 's' : ''}`
                )}
              </Button>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

export default UploadPDF
