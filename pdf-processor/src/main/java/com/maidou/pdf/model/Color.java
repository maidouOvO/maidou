package com.maidou.pdf.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

public class Color {
    private float red;
    private float green;
    private float blue;
    private float alpha;

    public Color() {}

    @JsonCreator
    public Color(
        @JsonProperty("red") float red,
        @JsonProperty("green") float green,
        @JsonProperty("blue") float blue,
        @JsonProperty("alpha") float alpha
    ) {
        this.red = red;
        this.green = green;
        this.blue = blue;
        this.alpha = alpha;
    }

    public Color(float red, float green, float blue) {
        this(red, green, blue, 1.0f);
    }

    public float getRed() { return red; }
    public void setRed(float red) { this.red = red; }
    public float getGreen() { return green; }
    public void setGreen(float green) { this.green = green; }
    public float getBlue() { return blue; }
    public void setBlue(float blue) { this.blue = blue; }
    public float getAlpha() { return alpha; }
    public void setAlpha(float alpha) { this.alpha = alpha; }
}
