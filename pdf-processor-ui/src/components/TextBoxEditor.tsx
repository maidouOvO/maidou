import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card"
import { Input } from "../components/ui/input"
import { Button } from "../components/ui/button"
import { Label } from "../components/ui/label"
import { ProcessedPDF } from '../lib/api'

interface TextBoxEditorProps {
  processedData: ProcessedPDF
  selectedBoxId?: number
  onUpdate: (boxId: number, updates: { text: string }) => void
}

export function TextBoxEditor({ processedData, selectedBoxId, onUpdate }: TextBoxEditorProps) {
  const [text, setText] = useState('')

  useEffect(() => {
    if (selectedBoxId && processedData) {
      const box = processedData.pages
        .flatMap(page => page.textBoxes)
        .find(box => box.id === selectedBoxId)
      if (box) {
        setText(box.content)
      }
    }
  }, [selectedBoxId, processedData])

  const handleUpdate = () => {
    if (selectedBoxId) {
      onUpdate(selectedBoxId, { text })
    }
  }

  if (!selectedBoxId) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Text Box Editor</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Select a text box to edit its content</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Text Box Editor - Box {selectedBoxId}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="text">Text Content</Label>
          <Input
            id="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text content"
          />
        </div>
        <Button onClick={handleUpdate} className="w-full">
          Update Text Box
        </Button>
      </CardContent>
    </Card>
  )
}
