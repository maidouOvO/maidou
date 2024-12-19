package com.maidou.pdf.storage;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.maidou.pdf.model.TextBox;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class MetadataStorage {
    private final ObjectMapper objectMapper;
    private final String outputDirectory;

    public MetadataStorage(String outputDirectory) {
        this.objectMapper = new ObjectMapper();
        this.outputDirectory = outputDirectory;

        // Create output directory if it doesn't exist
        new File(outputDirectory).mkdirs();
    }

    public void saveTextBoxMetadata(List<TextBox> textBoxes) throws IOException {
        // Group text boxes by book ID and page ID
        Map<String, Map<Integer, List<TextBox>>> bookPages = textBoxes.stream()
            .collect(Collectors.groupingBy(
                TextBox::getBookId,
                Collectors.groupingBy(TextBox::getPageId)
            ));

        // Save metadata for each book
        for (Map.Entry<String, Map<Integer, List<TextBox>>> bookEntry : bookPages.entrySet()) {
            String bookId = bookEntry.getKey();
            Map<Integer, List<TextBox>> pages = bookEntry.getValue();

            // Create book directory
            String bookDir = outputDirectory + File.separator + bookId;
            new File(bookDir).mkdirs();

            // Save each page's metadata
            for (Map.Entry<Integer, List<TextBox>> pageEntry : pages.entrySet()) {
                Integer pageId = pageEntry.getKey();
                List<TextBox> pageTextBoxes = pageEntry.getValue();

                String filename = String.format("%s/page_%d.json", bookDir, pageId);
                ObjectWriter writer = objectMapper.writerWithDefaultPrettyPrinter();
                writer.writeValue(new File(filename), pageTextBoxes);
            }
        }
    }

    public List<TextBox> loadTextBoxMetadata(String bookId, int pageId) throws IOException {
        String filename = String.format("%s/%s/page_%d.json", outputDirectory, bookId, pageId);
        return objectMapper.readValue(
            new File(filename),
            objectMapper.getTypeFactory().constructCollectionType(List.class, TextBox.class)
        );
    }
}
