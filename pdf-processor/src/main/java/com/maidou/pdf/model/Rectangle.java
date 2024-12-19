package com.maidou.pdf.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

public class Rectangle {
    private float x;
    private float y;
    private float width;
    private float height;

    // Default constructor for JSON deserialization
    public Rectangle() {}

    @JsonCreator
    public Rectangle(
        @JsonProperty("x") float x,
        @JsonProperty("y") float y,
        @JsonProperty("width") float width,
        @JsonProperty("height") float height
    ) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    // Getters and setters
    public float getX() { return x; }
    public void setX(float x) { this.x = x; }
    public float getY() { return y; }
    public void setY(float y) { this.y = y; }
    public float getWidth() { return width; }
    public void setWidth(float width) { this.width = width; }
    public float getHeight() { return height; }
    public void setHeight(float height) { this.height = height; }
}
