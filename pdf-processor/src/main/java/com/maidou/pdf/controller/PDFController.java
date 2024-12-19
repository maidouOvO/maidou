package com.maidou.pdf.controller;

import com.maidou.pdf.PDFProcessor;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.io.IOException;

@RestController
@RequestMapping("/api/pdf")
public class PDFController {
    private final PDFProcessor pdfProcessor;

    public PDFController(PDFProcessor pdfProcessor) {
        this.pdfProcessor = pdfProcessor;
    }

    @PostMapping("/process")
    public ResponseEntity<Resource> processPDF(
            @RequestParam("file") MultipartFile file,
            @RequestParam("bookId") String bookId) throws IOException {
        // Process PDF and get byte array
        byte[] pdfBytes = pdfProcessor.processPDFToBytes(file, bookId);

        ByteArrayResource resource = new ByteArrayResource(pdfBytes);

        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"processed.pdf\"")
                .contentType(MediaType.APPLICATION_PDF)
                .contentLength(pdfBytes.length)
                .body(resource);
    }
}
