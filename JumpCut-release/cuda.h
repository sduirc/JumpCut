#define XY_TO_INT(x, y) (((y)<<12)|(x))
#define INT_TO_X(vFG) ((vFG)&((1<<12)-1))
#define INT_TO_Y(vFG) ((vFG)>>12)

extern int global_patch_w;
extern int global_pm_iters;

double inline __declspec (naked) __fastcall sqrtNani(double n)
{
	_asm
	{
		fld qword ptr[esp + 4]
			fsqrt
			ret 8
	}
}