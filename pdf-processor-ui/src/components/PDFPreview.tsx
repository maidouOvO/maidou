import { useState } from 'react'
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ProcessedPDF } from '@/lib/api'
import { cn } from "@/lib/utils"

interface PDFPreviewProps {
  url: string
  processedData?: ProcessedPDF
  onTextBoxSelect?: (boxId: number) => void
}

export function PDFPreview({ url, processedData, onTextBoxSelect }: PDFPreviewProps) {
  const [scale, setScale] = useState(1)
  const [selectedBox, setSelectedBox] = useState<number | null>(null)

  const handleTextBoxClick = (boxId: number) => {
    setSelectedBox(boxId)
    onTextBoxSelect?.(boxId)
  }

  return (
    <Card className="p-4">
      <div className="flex justify-end gap-2 mb-4">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setScale(s => Math.max(0.5, s - 0.1))}
        >
          Zoom Out
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setScale(s => Math.min(2, s + 0.1))}
        >
          Zoom In
        </Button>
      </div>

      <div className="relative border rounded-lg overflow-hidden">
        <iframe
          src={url}
          className="w-full h-96 border-0"
          style={{ transform: `scale(${scale})`, transformOrigin: 'top left' }}
          title="PDF Preview"
        />

        {processedData?.pages.map(page => (
          <div key={page.pageNumber} className="absolute inset-0 pointer-events-none">
            {page.textBoxes.map(box => (
              <div
                key={box.id}
                className={cn(
                  "absolute border-2 cursor-pointer pointer-events-auto transition-colors",
                  selectedBox === box.id ? "border-blue-500 bg-blue-50/20" : "border-yellow-300"
                )}
                style={{
                  left: `${box.x}px`,
                  top: `${box.y}px`,
                  width: `${box.width}px`,
                  height: `${box.height}px`,
                  transform: `scale(${scale})`,
                  transformOrigin: 'top left'
                }}
                onClick={() => handleTextBoxClick(box.id)}
              >
                <span className="absolute -top-6 left-0 bg-white px-2 py-1 text-sm border rounded">
                  {box.id}
                </span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </Card>
  )
}
