#pragma once
class BITMAP3
{
public:
	BITMAP3();
	~BITMAP3();

	BITMAP3(int w_, int h_) :w(w_), h(h_) { data = new int[w*h]; }

	int w, h;
	int *data;
	int *operator[](int y) { return &data[y*w]; }
};