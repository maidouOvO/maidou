package com.maidou.pdf.controller;

import com.maidou.pdf.PDFProcessor;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mockito;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.junit4.SpringRunner;
import org.springframework.test.web.servlet.MockMvc;

import static org.mockito.ArgumentMatchers.any;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@RunWith(SpringRunner.class)
@WebMvcTest(PDFController.class)
public class PDFControllerTest {
    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private PDFProcessor pdfProcessor;

    @Test
    public void testProcessPDF() throws Exception {
        // Prepare test data
        byte[] pdfContent = "test pdf content".getBytes();
        MockMultipartFile file = new MockMultipartFile(
            "file",
            "test.pdf",
            MediaType.APPLICATION_PDF_VALUE,
            "test pdf content".getBytes()
        );
        String bookId = "test-book-id";

        // Mock PDFProcessor behavior
        Mockito.when(pdfProcessor.processPDFToBytes(any(), any())).thenReturn(pdfContent);

        // Perform request and verify response
        mockMvc.perform(multipart("/api/pdf/process")
                .file(file)
                .param("bookId", bookId))
                .andExpect(status().isOk())
                .andExpect(header().string(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_PDF_VALUE))
                .andExpect(header().string(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"processed.pdf\""))
                .andExpect(content().bytes(pdfContent));
    }
}
