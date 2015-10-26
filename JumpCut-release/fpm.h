
#ifndef _INC_FPM_H
#define _INC_FPM_H


#ifdef FPMLIB_EXPORT
#define _FPM_API __declspec(dllexport)
#else
#define _FPM_API __declspec(dllimport)
#endif


enum
{
	FPMT_BEG=0, FPMT_8U, FPMT_32S, FPMT_32F, FPMT_64F, FPMT_END
};

enum
{
	FPMF_WEIGHTED=0x01, FPMF_TRUE_SSD=0x02
};

class _FPM_API  IFPM
{
protected:
	
	/*IFPM(const IFPM &);
	IFPM& operator=(const IFPM &);*/

public:
	
	IFPM();

public:

	//plan the searching process.
	
	//@iwidth,@iheight: image size
	
	//@pwidth,@pheight: pattern size
	
	//@dim : the dimension of each element, typically, the channel number of image and pattern.
	
	//@flag: (FPMF_WEIGHTED)
	
	//@sel : a value to specify what and how many patches can be found.
	
	virtual void Plan(int iwidth, int iheight, /*int pwidth, int pheight, */int dim, int flag, double sel)=0;

	//set source image and search locations.
	
	//@img, @istep, @itype : the image data, step and data type, the data type can be any of FPMT_*
	
	//@mask, @mstep : mask of the search location, if @mask is NULL, all locations would be searched. Note that
	//				  the boundary elements with x>@iwidth-@pwidth or y>@iheight-@pheight would never be searched 
	//				  regardless their mask.
	
	virtual void SetImage(const void *img, int istep, int itype, const unsigned char *mask, int mstep) =0 ;

	//search a pattern in the pre-set image.
	
	//@pat,@pstep,@ptype : the pattern data, step and type.
	
	//@weight			 : the weight for each element in @pat. if it is planned to be weighted, @weight should not be null, else
	//						it would be ignored.
	
	//@pmatch            : output the indices of the matched locations. The related coordinates can be computed from an index i as (i%iwidth,i/iwidth)
	
	//@pssd				 : the SSD match error of each match location, it can be NULL.
	
	//@nmax				 : max matches to search.
	
	//@flag				 : FPMF_TRUE_SSD => output true SSD, otherwise the SSD in @pssd may be different from its true value with a constant offset.

	//return value : the number of matches be found.

	virtual int Match(const void *pat, int pwidth, int pheight, int pstep, int ptype, const double *weight, int *pmatch,double *pssd,int nmax,int flag) =0;

	virtual ~IFPM();
};

#if 1

class _FPM_API  FPMNaive
	:public IFPM
{
	class _CImp;

	_CImp *m_pImp;

public:

	FPMNaive();

	virtual ~FPMNaive();

	//get the SSD of all elements.
	
	//@pat,@pstep,@ptype,@weight is the same to those in @Match(...)
	
	//@pdelta : the constant offset of the result SSD to its true value, it can be NULL if not necessary.

	virtual const double* GetSSD(const void *pat, int pwidth, int pheight, int pstep, int ptype, const double *weight,double *pdelta);

public:

	// @sel : if SSD(p)<SSDmin+@sel, p would be a matched location, where SSD(p) is the SSD of the location p, SSDmin is the minimum SSD of all locations.

	virtual void Plan(int iwidth, int iheight, int dim,int flag, double sel);

	virtual void SetImage(const void *img, int istep,  int itype, const unsigned char *mask, int mstep) ;

	virtual int Match(const void *pat, int pwidth, int pheight, int pstep, int ptype, const double *weight, int *pmatch,double *pssd,int nmax,int flag) ;
};

#endif


class _FPM_API  FPM_FFT
	:public IFPM
{
	class _CImp;

	_CImp *m_pImp;

public:

	FPM_FFT();

	virtual ~FPM_FFT();

	// @sel : if SSD(p)<SSDmin+@sel, p would be a matched location, where SSD(p) is the SSD of the location p, SSDmin is the minimum SSD of all locations.

	virtual const double* GetSSD(const void *pat, int pwidth, int pheight, int pstep, int ptype, const double *weight,double *pdelta);

public:
	virtual void Plan(int iwidth, int iheight, /*int pwidth, int pheight,*/int dim, int flag, double sel);

	virtual void SetImage(const void *img, int istep, int itype, const unsigned char *mask, int mstep) ;

	virtual int Match(const void *pat, int pwidth, int pheight, int pstep, int ptype, const double *weight, int *pmatch,double *pssd,int nmax,int flag) ;
};


//make pattern-mask from pixel-mask.
//this function can be used to get the mask of seach location(pattern-mask) from a destroied image, e.g., the image when do image completion.
//the pixel-mask is the mask of pixels, with the unmasked pixels be destroied. The pattern-mask is made so that all the pixels covered by the
//pattern at a search location are valid, with boundary pixels excluded.

//@iwidth,@iheight,@pwidth,@pheight : the image and pattern size.

//@pPixelMask, @xmstep : pixel-mask and its step. If @pPixelMask is NULL, all pixels are valid and only boundary pixels are masked out.

//@pPatMask, @pmstep : pattern-mask and its step.

//@mval : mask value for valid pixels, which should not be zero.

void FPMMakePatMask(int iwidth, int iheight, int pwidth, int pheight, const unsigned char *pPixelMask, int xmstep, 
					unsigned char *pPatMask, int pmstep,
					unsigned char mval =1
					);






#endif

