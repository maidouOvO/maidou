package com.maidou.pdf;

import net.sourceforge.tess4j.*;
import com.maidou.pdf.model.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import org.springframework.web.multipart.MultipartFile;
import javax.imageio.ImageIO;
import java.io.IOException;

/**
 * OCR处理器 - 用于从图片中提取文字
 */
public class OCRProcessor {
    private final Tesseract tesseract;

    public OCRProcessor() {
        tesseract = new Tesseract();
        tesseract.setLanguage("chi_sim+eng");  // 支持中文简体和英文
    }

    /**
     * 从MultipartFile中提取文字并创建TextBox列表
     * @param file 要处理的图片文件
     * @param bookId 书籍ID
     * @param pageId 页面ID
     * @return 包含提取文字的TextBox列表
     * @throws TesseractException 如果OCR处理失败
     * @throws IOException 如果文件处理失败
     */
    public List<TextBox> extractTextFromFile(MultipartFile file, String bookId, int pageId) throws TesseractException, IOException {
        BufferedImage image = ImageIO.read(file.getInputStream());
        return extractTextFromImage(image, bookId, pageId);
    }

    /**
     * 从图片中提取文字并创建TextBox列表
     * @param image 要处理的图片
     * @param bookId 书籍ID
     * @param pageId 页面ID
     * @return 包含提取文字的TextBox列表
     * @throws TesseractException 如果OCR处理失败
     */
    public List<TextBox> extractTextFromImage(BufferedImage image, String bookId, int pageId) throws TesseractException {
        List<TextBox> textBoxes = new ArrayList<>();
        List<Word> words = tesseract.getWords(image, ITessAPI.TessPageIteratorLevel.RIL_WORD);
        for (Word word : words) {
            Rectangle rect = new Rectangle(
                word.getBoundingBox().x,
                word.getBoundingBox().y,
                word.getBoundingBox().width,
                word.getBoundingBox().height
            );

            TextBox textBox = new TextBox(
                bookId,
                pageId,
                java.util.UUID.randomUUID().toString(),
                rect,
                new Color(0, 0, 0, 1.0f),  // 默认黑色文字
                TextAlignment.LEFT,         // 默认左对齐
                word.getText(),
                0  // 序号将在PDFProcessor中设置
            );
            textBoxes.add(textBox);
        }
        return textBoxes;
    }
}
