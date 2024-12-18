import { useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { FileUp, X, AlertCircle, Download } from "lucide-react"
import { useDropzone } from 'react-dropzone'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { uploadPDF, exportPDF, exportTable, type ProcessedPDF } from './lib/api'
import { PDFPreview } from './components/PDFPreview'
import { TextBoxEditor } from './components/TextBoxEditor'

function App() {
  const [pdfFile, setPdfFile] = useState<File | null>(null)
  const [width, setWidth] = useState<string>("800")
  const [height, setHeight] = useState<string>("1200")
  const [previewUrl, setPreviewUrl] = useState<string>("")
  const [error, setError] = useState<string>("")
  const [isProcessing, setIsProcessing] = useState(false)
  const [processedData, setProcessedData] = useState<ProcessedPDF | undefined>(undefined)
  const [selectedBoxId, setSelectedBoxId] = useState<number | undefined>()

  const validateDimensions = (w: string, h: string) => {
    const numWidth = parseInt(w)
    const numHeight = parseInt(h)
    if (isNaN(numWidth) || isNaN(numHeight)) {
      setError("Width and height must be numbers")
      return false
    }
    if (numWidth < 1 || numHeight < 1) {
      setError("Width and height must be greater than 0")
      return false
    }
    if (numWidth > 5000 || numHeight > 5000) {
      setError("Width and height must be less than 5000 pixels")
      return false
    }
    setError("")
    return true
  }

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file?.type === 'application/pdf') {
      setPdfFile(file)
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
    } else {
      alert('Please upload a PDF file')
    }
  }, [])

  const clearFile = useCallback(() => {
    setPdfFile(null)
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
      setPreviewUrl("")
    }
  }, [previewUrl])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: false
  })

  const handleTextBoxSelect = (boxId: number) => {
    setSelectedBoxId(boxId)
    console.log('Selected text box:', boxId)
  }

  const handleTextBoxUpdate = (boxId: number, updates: { text: string }) => {
    if (!processedData) return

    const updatedData: ProcessedPDF = {
      ...processedData,
      pages: processedData.pages.map(page => ({
        ...page,
        textBoxes: page.textBoxes.map(box =>
          box.id === boxId ? { ...box, ...updates } : box
        )
      }))
    }
    setProcessedData(updatedData)
  }

  const handleProcess = async () => {
    if (!pdfFile || !width || !height) return

    try {
      setIsProcessing(true)
      setError("")
      const data = await uploadPDF(pdfFile, parseInt(width), parseInt(height))
      setProcessedData(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process PDF')
      setProcessedData(undefined)
    } finally {
      setIsProcessing(false)
    }
  }

  const handleExportPDF = async () => {
    if (!processedData) return

    try {
      const blob = await exportPDF(processedData.bookId)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${processedData.bookId}_${processedData.bookName}_processed.pdf`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to export PDF')
    }
  }

  const handleExportTable = async () => {
    if (!processedData) return

    try {
      const blob = await exportTable(processedData.bookId)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${processedData.bookId}_${processedData.bookName}_results.csv`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to export table')
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-center text-gray-900">PDF Text Box Processor</h1>
        <p className="text-center text-gray-600 mt-2">Upload a PDF file and set dimensions to process text boxes</p>
      </header>

      <main className="container mx-auto max-w-4xl">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              Upload PDF
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <AlertCircle className="h-5 w-5 text-gray-400" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Upload a PDF file to process text boxes</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-6">
              {error && (
                <Alert variant="destructive">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              <div
                {...getRootProps()}
                className={`relative border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors
                  ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}`}
              >
                <input {...getInputProps()} />
                {pdfFile && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="absolute top-2 right-2"
                    onClick={(e) => {
                      e.stopPropagation()
                      clearFile()
                    }}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                )}
                <FileUp className="mx-auto h-12 w-12 text-gray-400" />
                <div className="mt-4">
                  {pdfFile ? (
                    <p className="text-sm text-gray-600">Selected: {pdfFile.name}</p>
                  ) : (
                    <p className="text-sm text-gray-600">
                      {isDragActive ? 'Drop the PDF here' : 'Drag and drop your PDF here, or click to select'}
                    </p>
                  )}
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                {previewUrl && (
                  <>
                    <PDFPreview
                      url={previewUrl}
                      processedData={processedData}
                      onTextBoxSelect={handleTextBoxSelect}
                    />
                    {processedData && (
                      <TextBoxEditor
                        processedData={processedData}
                        selectedBoxId={selectedBoxId}
                        onUpdate={handleTextBoxUpdate}
                      />
                    )}
                  </>
                )}
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Width (pixels)
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger className="ml-1">
                          <AlertCircle className="inline h-4 w-4 text-gray-400" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>Enter the desired width in pixels (1-5000)</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </label>
                  <Input
                    type="number"
                    value={width}
                    onChange={(e) => {
                      setWidth(e.target.value)
                      validateDimensions(e.target.value, height)
                    }}
                    placeholder="Enter width"
                    min="1"
                    max="5000"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Height (pixels)
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger className="ml-1">
                          <AlertCircle className="inline h-4 w-4 text-gray-400" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>Enter the desired height in pixels (1-5000)</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </label>
                  <Input
                    type="number"
                    value={height}
                    onChange={(e) => {
                      setHeight(e.target.value)
                      validateDimensions(width, e.target.value)
                    }}
                    placeholder="Enter height"
                    min="1"
                    max="5000"
                  />
                </div>
              </div>

              <div className="flex gap-4">
                <Button
                  className="flex-1"
                  disabled={!pdfFile || !width || !height || !!error || isProcessing}
                  onClick={handleProcess}
                >
                  {isProcessing ? 'Processing...' : 'Process PDF'}
                </Button>

                {processedData && (
                  <>
                    <Button
                      variant="outline"
                      onClick={handleExportPDF}
                      className="flex items-center gap-2"
                    >
                      <Download className="h-4 w-4" />
                      Export PDF
                    </Button>
                    <Button
                      variant="outline"
                      onClick={handleExportTable}
                      className="flex items-center gap-2"
                    >
                      <Download className="h-4 w-4" />
                      Export Table
                    </Button>
                  </>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  )
}

export default App
