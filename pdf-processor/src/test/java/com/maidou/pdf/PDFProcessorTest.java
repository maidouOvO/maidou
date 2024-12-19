package com.maidou.pdf;

import com.maidou.pdf.model.*;
import com.maidou.pdf.storage.MetadataStorage;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.font.PDType1Font;
import org.junit.Before;
import org.junit.Test;
import org.junit.After;
import org.junit.Rule;
import org.junit.rules.TemporaryFolder;
import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;
import java.util.List;

public class PDFProcessorTest {
    private static final String TEST_BOOK_ID = "test-book";
    private PDFProcessor processor;
    private PDDocument document;

    @Rule
    public TemporaryFolder tempFolder = new TemporaryFolder();

    @Before
    public void setUp() throws IOException {
        processor = new PDFProcessor(TEST_BOOK_ID);
        document = createTestPDF();
    }

    @After
    public void tearDown() throws IOException {
        if (document != null) {
            document.close();
        }
    }

    private PDDocument createTestPDF() throws IOException {
        PDDocument doc = new PDDocument();
        PDPage page = new PDPage();
        doc.addPage(page);

        try (PDPageContentStream contentStream = new PDPageContentStream(doc, page)) {
            // Add test text
            contentStream.beginText();
            contentStream.setFont(PDType1Font.HELVETICA, 12);
            contentStream.newLineAtOffset(100, 700);
            contentStream.showText("Test Content Line 1");
            contentStream.newLineAtOffset(0, -20);
            contentStream.showText("Test Content Line 2");
            contentStream.endText();

            // Add a simple shape that will be detected as an image
            contentStream.setLineWidth(2);
            contentStream.addRect(200, 650, 100, 100);
            contentStream.stroke();
        }

        return doc;
    }

    @Test
    public void testTextExtraction() throws IOException {
        List<TextBox> textBoxes = processor.processPage(document, 1);

        assertNotNull("Text boxes should not be null", textBoxes);
        assertFalse("Text boxes should not be empty", textBoxes.isEmpty());
        assertTrue("Should find at least 2 text boxes", textBoxes.size() >= 2);

        TextBox firstBox = textBoxes.get(0);
        assertEquals("Book ID should match", TEST_BOOK_ID, firstBox.getBookId());
        assertEquals("Page ID should be 1", 1, firstBox.getPageId());
        assertNotNull("Text box ID should not be null", firstBox.getTextBoxId());
        assertTrue("Content should contain test text",
            firstBox.getContent().contains("Test Content"));
        assertEquals("First box should have order number 1", 1, firstBox.getOrderNumber());
    }

    @Test
    public void testImageExtraction() throws IOException {
        List<BufferedImage> images = processor.extractImages(document);
        assertNotNull("Images list should not be null", images);

        // Our test PDF contains a rectangle shape
        // While it's not a true image, it should be detected as a vector graphic
        assertFalse("Vector graphics list should not be empty", images.isEmpty());
    }

    @Test
    public void testYellowBoxRendering() throws IOException {
        // Process the page which adds yellow boxes
        List<TextBox> textBoxes = processor.processPage(document, 1);

        // Save the modified PDF to verify the changes were made
        File outputFile = tempFolder.newFile("test-output.pdf");
        document.save(outputFile);

        assertTrue("Modified PDF should exist", outputFile.exists());
        assertTrue("Modified PDF should have content", outputFile.length() > 0);

        // Verify text box properties
        TextBox firstBox = textBoxes.get(0);
        Rectangle coords = firstBox.getCoordinates();

        assertNotNull("Coordinates should not be null", coords);
        assertTrue("Width should be positive", coords.getWidth() > 0);
        assertTrue("Height should be positive", coords.getHeight() > 0);
        assertTrue("X coordinate should be around 100",
            Math.abs(coords.getX() - 100) < 10);
        assertTrue("Y coordinate should be around 700",
            Math.abs(coords.getY() - 700) < 10);
    }

    @Test
    public void testMetadataStorage() throws IOException {
        // Process the page
        List<TextBox> textBoxes = processor.processPage(document, 1);

        // Create metadata storage
        File storageDir = tempFolder.newFolder("metadata");
        MetadataStorage storage = new MetadataStorage(storageDir.getAbsolutePath());

        // Save and load metadata
        storage.saveTextBoxMetadata(textBoxes);
        List<TextBox> loadedBoxes = storage.loadTextBoxMetadata(TEST_BOOK_ID, 1);

        assertEquals("Number of text boxes should match",
            textBoxes.size(), loadedBoxes.size());

        // Compare first text box
        TextBox original = textBoxes.get(0);
        TextBox loaded = loadedBoxes.get(0);
        assertEquals("Content should match",
            original.getContent(), loaded.getContent());
        assertEquals("Order number should match",
            original.getOrderNumber(), loaded.getOrderNumber());
    }
}
