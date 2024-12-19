package com.maidou.pdf;

import com.maidou.pdf.model.TextBox;
import com.maidou.pdf.storage.MetadataStorage;
import org.apache.pdfbox.pdmodel.PDDocument;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;
import javax.imageio.ImageIO;

public class Main {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Usage: java -jar pdf-processor.jar <input-pdf-path> [output-dir]");
            System.exit(1);
        }

        String inputPath = args[0];
        String outputDir = args.length > 1 ? args[1] : "output";
        String bookId = new File(inputPath).getName().replaceAll("\\.pdf$", "");

        try {
            // 创建输出目录
            new File(outputDir).mkdirs();

            // 初始化处理器和存储
            PDFProcessor processor = new PDFProcessor();
            MetadataStorage storage = new MetadataStorage(outputDir);

            // 打开PDF文件
            File pdfFile = new File(inputPath);
            try (PDDocument document = PDDocument.load(pdfFile)) {
                System.out.println("Processing PDF: " + inputPath);
                System.out.println("Total pages: " + document.getNumberOfPages());

                // 处理每一页
                for (int i = 1; i <= document.getNumberOfPages(); i++) {
                    System.out.println("\nProcessing page " + i);

                    // 提取和处理文本框
                    List<TextBox> textBoxes = processor.processPage(document, i, bookId);
                    System.out.println("Found " + textBoxes.size() + " text boxes");

                    // 保存文本框元数据
                    storage.saveTextBoxMetadata(textBoxes);

                    // 输出文本框信息
                    for (TextBox box : textBoxes) {
                        System.out.printf("Text Box %d: %s%n",
                            box.getOrderNumber(),
                            box.getContent().substring(0, Math.min(50, box.getContent().length())));
                    }
                }

                // 提取图片
                List<BufferedImage> images = processor.extractImages(document);
                System.out.println("\nFound " + images.size() + " images");

                // 保存图片
                for (int i = 0; i < images.size(); i++) {
                    File imageFile = new File(outputDir, String.format("%s_image_%d.png", bookId, i + 1));
                    ImageIO.write(images.get(i), "PNG", imageFile);
                    System.out.println("Saved image: " + imageFile.getName());
                }

                // 保存处理后的PDF
                File outputFile = new File(outputDir, bookId + "_processed.pdf");
                document.save(outputFile);
                System.out.println("\nProcessed PDF saved to: " + outputFile.getAbsolutePath());
            }

        } catch (IOException e) {
            System.err.println("Error processing PDF: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
