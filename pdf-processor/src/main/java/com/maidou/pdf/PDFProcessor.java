package com.maidou.pdf;

import com.maidou.pdf.model.*;
import org.apache.pdfbox.cos.COSName;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.PDResources;
import org.apache.pdfbox.pdmodel.common.PDRectangle;
import org.apache.pdfbox.pdmodel.font.PDType1Font;
import org.apache.pdfbox.pdmodel.graphics.form.PDFormXObject;
import org.apache.pdfbox.pdmodel.graphics.image.PDImageXObject;
import org.apache.pdfbox.pdmodel.graphics.PDXObject;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.pdfbox.text.TextPosition;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class PDFProcessor {
    private static final Color LEMON_YELLOW = new Color(1.0f, 1.0f, 0.0f, 0.3f); // Normalized RGB values with 30% transparency
    private static final float BORDER_WIDTH = 2.0f;
    private static final float BORDER_RADIUS = 5.0f;
    private String currentBookId;
    private PDDocument document;

    public PDFProcessor(String bookId) {
        this.currentBookId = bookId;
    }

    public List<TextBox> processPage(PDDocument document, int pageNumber) throws IOException {
        this.document = document;
        PDPage page = document.getPage(pageNumber - 1);
        List<TextBox> textBoxes = extractTextBoxes(page, pageNumber);

        // Sort text boxes from top to bottom, left to right
        textBoxes.sort((a, b) -> {
            Rectangle aCoord = a.getCoordinates();
            Rectangle bCoord = b.getCoordinates();
            if (Math.abs(aCoord.getY() - bCoord.getY()) > 10) {
                return Float.compare(bCoord.getY(), aCoord.getY()); // Higher Y is at top
            }
            return Float.compare(aCoord.getX(), bCoord.getX());
        });

        // Assign order numbers
        for (int i = 0; i < textBoxes.size(); i++) {
            textBoxes.get(i).setOrderNumber(i + 1);
        }

        // Add yellow borders
        try (PDPageContentStream contentStream = new PDPageContentStream(document, page, PDPageContentStream.AppendMode.APPEND, true, true)) {
            for (TextBox textBox : textBoxes) {
                drawYellowBorder(contentStream, textBox);
                drawOrderNumber(contentStream, textBox);
            }
        }

        return textBoxes;
    }

    private List<TextBox> extractTextBoxes(PDPage page, int pageNumber) throws IOException {
        CustomPDFTextStripper stripper = new CustomPDFTextStripper();
        stripper.setStartPage(pageNumber);
        stripper.setEndPage(pageNumber);
        stripper.getText(document);
        return stripper.getTextBoxes();
    }

    private void drawYellowBorder(PDPageContentStream contentStream, TextBox textBox) throws IOException {
        Rectangle coords = textBox.getCoordinates();
        contentStream.setStrokingColor(1.0f, 1.0f, 0.0f); // Normalized RGB values for yellow
        contentStream.setLineWidth(BORDER_WIDTH);

        // Draw rounded rectangle
        float x = coords.getX();
        float y = coords.getY();
        float width = coords.getWidth();
        float height = coords.getHeight();

        contentStream.moveTo(x + BORDER_RADIUS, y);
        contentStream.lineTo(x + width - BORDER_RADIUS, y);
        contentStream.curveTo(x + width, y, x + width, y, x + width, y + BORDER_RADIUS);
        contentStream.lineTo(x + width, y + height - BORDER_RADIUS);
        contentStream.curveTo(x + width, y + height, x + width, y + height, x + width - BORDER_RADIUS, y + height);
        contentStream.lineTo(x + BORDER_RADIUS, y + height);
        contentStream.curveTo(x, y + height, x, y + height, x, y + height - BORDER_RADIUS);
        contentStream.lineTo(x, y + BORDER_RADIUS);
        contentStream.curveTo(x, y, x, y, x + BORDER_RADIUS, y);
        contentStream.stroke();
    }

    private void drawOrderNumber(PDPageContentStream contentStream, TextBox textBox) throws IOException {
        Rectangle coords = textBox.getCoordinates();
        contentStream.beginText();
        contentStream.setFont(PDType1Font.HELVETICA_BOLD, 10);
        contentStream.newLineAtOffset(coords.getX() + 2, coords.getY() + coords.getHeight() - 12);
        contentStream.showText(String.valueOf(textBox.getOrderNumber()));
        contentStream.endText();
    }

    public List<BufferedImage> extractImages(PDDocument document) throws IOException {
        List<BufferedImage> images = new ArrayList<>();
        PDFRenderer renderer = new PDFRenderer(document);

        for (PDPage page : document.getPages()) {
            // First, get any raster images
            PDResources resources = page.getResources();
            for (COSName name : resources.getXObjectNames()) {
                PDXObject xobject = resources.getXObject(name);
                if (xobject instanceof PDImageXObject) {
                    PDImageXObject image = (PDImageXObject) xobject;
                    images.add(image.getImage());
                }
            }

            // Then render the entire page to capture vector graphics
            BufferedImage pageImage = renderer.renderImageWithDPI(document.getPages().indexOf(page), 300);
            if (pageImage != null) {
                images.add(pageImage);
            }
        }
        return images;
    }

    private class CustomPDFTextStripper extends PDFTextStripper {
        private List<TextBox> textBoxes;
        private StringBuilder currentText;
        private List<TextPosition> textPositions;
        private float startX;
        private float lastY;
        private int orderNumber;

        public CustomPDFTextStripper() throws IOException {
            textBoxes = new ArrayList<>();
            currentText = new StringBuilder();
            textPositions = new ArrayList<>();
            startX = -1;
            lastY = -1;
            orderNumber = 1;
        }

        @Override
        protected void writeString(String text, List<TextPosition> positions) throws IOException {
            if (positions.isEmpty()) return;

            TextPosition firstPosition = positions.get(0);
            float currentY = firstPosition.getY();

            // Start new text box if Y position changes significantly
            if (lastY != -1 && Math.abs(currentY - lastY) > 5) {
                if (currentText.length() > 0) {
                    createTextBox();
                }
                startNewTextBox(firstPosition);
                this.textPositions.clear();
            }

            // Initialize start position for new text box
            if (startX == -1) {
                startNewTextBox(firstPosition);
            }

            currentText.append(text);
            this.textPositions.addAll(positions);
            lastY = currentY;
        }

        private void startNewTextBox(TextPosition position) {
            startX = position.getX();
            currentText = new StringBuilder();
            textPositions = new ArrayList<>();
        }

        private void createTextBox() {
            if (currentText.length() == 0 || textPositions.isEmpty()) return;

            // Calculate text box coordinates
            float minX = startX;
            float minY = Float.MAX_VALUE;
            float maxX = minX;
            float maxY = Float.MIN_VALUE;

            for (TextPosition pos : textPositions) {
                maxX = Math.max(maxX, pos.getX() + pos.getWidth());
                minY = Math.min(minY, pos.getY());
                maxY = Math.max(maxY, pos.getY() + pos.getHeight());
            }

            // Adjust Y coordinate to match PDF coordinate system (origin at bottom-left)
            float pageHeight = getCurrentPage().getMediaBox().getHeight();
            float height = maxY - minY;
            minY = pageHeight - maxY;
            maxY = minY + height;

            Rectangle coordinates = new Rectangle(
                minX,
                minY,
                maxX - minX,
                maxY - minY
            );

            TextBox textBox = new TextBox(
                currentBookId,
                getCurrentPageNo(),
                UUID.randomUUID().toString(),
                coordinates,
                new Color(0, 0, 0, 1.0f),
                determineAlignment(textPositions),
                currentText.toString(),
                orderNumber++
            );

            textBoxes.add(textBox);
        }

        private TextAlignment determineAlignment(List<TextPosition> positions) {
            if (positions.isEmpty()) return TextAlignment.LEFT;

            // Simple alignment detection based on position relative to page width
            PDRectangle pageSize = document.getPage(getCurrentPageNo() - 1).getMediaBox();
            float pageWidth = pageSize.getWidth();
            float textStart = positions.get(0).getX();
            float margin = 50; // Typical margin size

            if (textStart < margin) return TextAlignment.LEFT;
            if (Math.abs(textStart - (pageWidth / 2)) < margin) return TextAlignment.CENTER;
            return TextAlignment.RIGHT;
        }

        public List<TextBox> getTextBoxes() {
            if (currentText.length() > 0) {
                createTextBox();
            }
            return textBoxes;
        }
    }
}
