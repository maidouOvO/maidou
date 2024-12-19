package com.maidou.pdf.model;

public class TextBox {
    private String bookId;
    private int pageId;
    private String textBoxId;
    private Rectangle coordinates;
    private Color textColor;
    private TextAlignment alignment;
    private String content;
    private int orderNumber;

    // Default constructor for JSON deserialization
    public TextBox() {}

    public TextBox(String bookId, int pageId, String textBoxId, Rectangle coordinates,
                  Color textColor, TextAlignment alignment, String content, int orderNumber) {
        this.bookId = bookId;
        this.pageId = pageId;
        this.textBoxId = textBoxId;
        this.coordinates = coordinates;
        this.textColor = textColor;
        this.alignment = alignment;
        this.content = content;
        this.orderNumber = orderNumber;
    }

    // Getters and setters
    public String getBookId() { return bookId; }
    public void setBookId(String bookId) { this.bookId = bookId; }
    public int getPageId() { return pageId; }
    public void setPageId(int pageId) { this.pageId = pageId; }
    public String getTextBoxId() { return textBoxId; }
    public void setTextBoxId(String textBoxId) { this.textBoxId = textBoxId; }
    public Rectangle getCoordinates() { return coordinates; }
    public void setCoordinates(Rectangle coordinates) { this.coordinates = coordinates; }
    public Color getTextColor() { return textColor; }
    public void setTextColor(Color textColor) { this.textColor = textColor; }
    public TextAlignment getAlignment() { return alignment; }
    public void setAlignment(TextAlignment alignment) { this.alignment = alignment; }
    public String getContent() { return content; }
    public void setContent(String content) { this.content = content; }
    public int getOrderNumber() { return orderNumber; }
    public void setOrderNumber(int orderNumber) { this.orderNumber = orderNumber; }
}
