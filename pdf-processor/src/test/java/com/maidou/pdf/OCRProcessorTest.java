package com.maidou.pdf;

import org.junit.Test;
import static org.junit.Assert.*;
import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.awt.Color;
import java.awt.Font;
import java.util.List;
import com.maidou.pdf.model.TextBox;
import net.sourceforge.tess4j.TesseractException;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.web.multipart.MultipartFile;
import javax.imageio.ImageIO;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

/**
 * OCR处理器测试类
 */
public class OCRProcessorTest {

    @Test
    public void testExtractTextFromFile() throws TesseractException, IOException {
        OCRProcessor processor = new OCRProcessor();

        // 创建测试图片
        BufferedImage testImage = createTestImage("测试文本 Test Text");

        // 将BufferedImage转换为MultipartFile
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(testImage, "png", baos);
        MultipartFile file = new MockMultipartFile(
            "test.png",
            "test.png",
            "image/png",
            baos.toByteArray()
        );

        // 测试文件处理
        List<TextBox> textBoxes = processor.extractTextFromFile(file, "test-book", 1);

        // 验证结果
        assertNotNull("文本框列表不应为空", textBoxes);
        assertFalse("文本框列表不应为空", textBoxes.isEmpty());

        // 验证文本框属性
        TextBox firstBox = textBoxes.get(0);
        assertNotNull("文本框坐标不应为空", firstBox.getCoordinates());
        assertNotNull("文本框内容不应为空", firstBox.getContent());
        assertEquals("书籍ID应匹配", "test-book", firstBox.getBookId());
        assertEquals("页面ID应匹配", 1, firstBox.getPageId());
    }

    /**
     * 创建包含文本的测试图片
     */
    private BufferedImage createTestImage(String text) {
        BufferedImage image = new BufferedImage(400, 100, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();

        // 设置白色背景
        g2d.setColor(Color.WHITE);
        g2d.fillRect(0, 0, image.getWidth(), image.getHeight());

        // 设置黑色文本
        g2d.setColor(Color.BLACK);
        g2d.setFont(new Font("SimSun", Font.PLAIN, 24)); // 使用支持中文的字体

        // 在图片中央绘制文本
        int x = 50;
        int y = 60;
        g2d.drawString(text, x, y);

        g2d.dispose();
        return image;
    }
}
