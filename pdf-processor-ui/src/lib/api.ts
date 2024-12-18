// API endpoints for PDF processing
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export interface ProcessedPDF {
  bookId: string
  bookName: string
  pages: Array<{
    pageNumber: number
    side: 'left' | 'right'
    textBoxes: Array<{
      id: number
      content: string
      x: number
      y: number
      width: number
      height: number
      color: string
      alignment: string
    }>
  }>
}

export const uploadPDF = async (
  file: File,
  width: number,
  height: number
): Promise<ProcessedPDF> => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('width', width.toString())
  formData.append('height', height.toString())

  const response = await fetch(`${API_BASE_URL}/process-pdf`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw new Error('Failed to process PDF')
  }

  return response.json()
}

export const exportPDF = async (bookId: string): Promise<Blob> => {
  const response = await fetch(`${API_BASE_URL}/export-pdf/${bookId}`)

  if (!response.ok) {
    throw new Error('Failed to export PDF')
  }

  return response.blob()
}

export const exportTable = async (bookId: string): Promise<Blob> => {
  const response = await fetch(`${API_BASE_URL}/export-table/${bookId}`)

  if (!response.ok) {
    throw new Error('Failed to export table')
  }

  return response.blob()
}
