import React, { useState, useEffect } from 'react'
import { Download, CheckCircle, AlertCircle, Clock, File, Package } from 'lucide-react'
import { Button } from './ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/Card'
import { Progress } from './ui/Progress'
import { formatDate } from '../lib/utils'

const FileProgressList = ({ taskId, files = [], onDownloadAll, onDownloadSingle }) => {
  const [expandedFiles, setExpandedFiles] = useState(new Set())
  const [overallProgress, setOverallProgress] = useState(0)

  useEffect(() => {
    if (files.length > 0) {
      const completedCount = files.filter(f => f.status === 'completed').length
      const progress = (completedCount / files.length) * 100
      setOverallProgress(progress)
    }
  }, [files])

  const toggleFileExpansion = (fileId) => {
    setExpandedFiles(prev => {
      const newSet = new Set(prev)
      if (newSet.has(fileId)) {
        newSet.delete(fileId)
      } else {
        newSet.add(fileId)
      }
      return newSet
    })
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'processing':
        return <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
      case 'failed':
        return <AlertCircle className="h-5 w-5 text-red-500" />
      case 'pending':
        return <Clock className="h-5 w-5 text-yellow-500" />
      default:
        return <Clock className="h-5 w-5 text-gray-400" />
    }
  }

  const getStatusBadge = (status) => {
    const baseClasses = "status-badge"
    switch (status) {
      case 'completed':
        return `${baseClasses} status-completed`
      case 'processing':
        return `${baseClasses} status-processing`
      case 'failed':
        return `${baseClasses} status-failed`
      case 'pending':
        return `${baseClasses} status-pending`
      default:
        return `${baseClasses} bg-gray-100 text-gray-800`
    }
  }

  const completedFiles = files.filter(f => f.status === 'completed')
  const allCompleted = files.length > 0 && completedFiles.length === files.length
  const hasCompletedFiles = completedFiles.length > 0

  if (!taskId || files.length === 0) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <File className="h-5 w-5" />
            <span>Processing Progress</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <File className="h-12 w-12 mx-auto mb-4 text-gray-300" />
            <p>No files being processed</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <File className="h-5 w-5" />
            <span>Processing Progress</span>
          </CardTitle>
          
          {hasCompletedFiles && (
            <Button
              onClick={() => onDownloadAll(taskId)}
              className="flex items-center space-x-2"
            >
              <Package className="h-4 w-4" />
              <span>Download All</span>
            </Button>
          )}
        </div>
        
        {/* Overall Progress */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">
              {completedFiles.length} of {files.length} files completed
            </span>
            <span className="text-gray-600">
              {Math.round(overallProgress)}%
            </span>
          </div>
          <Progress value={overallProgress} className="h-2" />
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {files.map((file, index) => (
          <div key={file.id || index} className="border border-gray-200 rounded-lg">
            <div
              className="p-4 cursor-pointer hover:bg-gray-50 transition-colors"
              onClick={() => toggleFileExpansion(file.id || index)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3 flex-1 min-w-0">
                  {getStatusIcon(file.status)}
                  
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {file.filename || file.name || `File ${index + 1}`}
                    </p>
                    <div className="flex items-center space-x-2 mt-1">
                      <span className={getStatusBadge(file.status)}>
                        {file.status}
                      </span>
                      
                      {file.processing_started && (
                        <span className="text-xs text-gray-500">
                          Started: {formatDate(file.processing_started)}
                        </span>
                      )}
                      
                      {file.processing_completed && (
                        <span className="text-xs text-gray-500">
                          Completed: {formatDate(file.processing_completed)}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  {file.status === 'completed' && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        onDownloadSingle(file.id, file.filename || file.name)
                      }}
                      className="flex items-center space-x-1"
                    >
                      <Download className="h-4 w-4" />
                      <span>Download</span>
                    </Button>
                  )}
                  
                  <div className="text-gray-400">
                    <svg
                      className={`h-5 w-5 transition-transform ${
                        expandedFiles.has(file.id || index) ? 'rotate-180' : ''
                      }`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Expanded Details */}
            {expandedFiles.has(file.id || index) && (
              <div className="px-4 pb-4 border-t border-gray-100 bg-gray-50">
                <div className="space-y-3 pt-3">
                  {/* Progress Bar */}
                  {file.status === 'processing' && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs text-gray-600">
                        <span>Processing...</span>
                        <span>{file.progress || 0}%</span>
                      </div>
                      <Progress value={file.progress || 0} className="h-1.5" />
                    </div>
                  )}
                  
                  {/* Error Message */}
                  {file.status === 'failed' && file.error_message && (
                    <div className="bg-red-50 border border-red-200 rounded-md p-3">
                      <p className="text-sm text-red-800">
                        <strong>Error:</strong> {file.error_message}
                      </p>
                    </div>
                  )}
                  
                  {/* File Details */}
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600">Original Filename:</span>
                      <p className="font-medium">{file.original_filename || file.filename}</p>
                    </div>
                    
                    {file.placeholder_count !== undefined && (
                      <div>
                        <span className="text-gray-600">Placeholders:</span>
                        <p className="font-medium">{file.placeholder_count}</p>
                      </div>
                    )}
                    
                    {file.created_at && (
                      <div>
                        <span className="text-gray-600">Uploaded:</span>
                        <p className="font-medium">{formatDate(file.created_at)}</p>
                      </div>
                    )}
                    
                    {file.updated_at && (
                      <div>
                        <span className="text-gray-600">Last Updated:</span>
                        <p className="font-medium">{formatDate(file.updated_at)}</p>
                      </div>
                    )}
                  </div>
                  
                  {/* Download Button for Completed Files */}
                  {file.status === 'completed' && (
                    <div className="pt-2">
                      <Button
                        onClick={() => onDownloadSingle(file.id, file.filename || file.name)}
                        className="w-full flex items-center justify-center space-x-2"
                      >
                        <Download className="h-4 w-4" />
                        <span>Download Generated Document</span>
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}
        
        {/* Summary */}
        {files.length > 0 && (
          <div className="pt-4 border-t border-gray-200">
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-1">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span className="text-gray-600">
                    {completedFiles.length} completed
                  </span>
                </div>
                
                {files.filter(f => f.status === 'processing').length > 0 && (
                  <div className="flex items-center space-x-1">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                    <span className="text-gray-600">
                      {files.filter(f => f.status === 'processing').length} processing
                    </span>
                  </div>
                )}
                
                {files.filter(f => f.status === 'failed').length > 0 && (
                  <div className="flex items-center space-x-1">
                    <AlertCircle className="h-4 w-4 text-red-500" />
                    <span className="text-gray-600">
                      {files.filter(f => f.status === 'failed').length} failed
                    </span>
                  </div>
                )}
              </div>
              
              {allCompleted && (
                <div className="text-green-600 font-medium">
                  All files completed successfully!
                </div>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default FileProgressList
