#include <iostream>
#include <windows.h>

//// 函数指针类型的定义，用于保存 DLL 中函数的地址
//typedef int (*START)();
//
//int main11()
//{
//	// 加载 DLL
//	HINSTANCE hinstDLL = LoadLibrary(TEXT("KuaiZai.dll"));
//
//	// 判断 DLL 是否加载成功
//	if (hinstDLL != NULL)
//	{
//		// 获取 DLL 中函数的地址
//		START pMultiply = (START)GetProcAddress(hinstDLL, "start");
//
//		// 判断函数地址是否获取成功
//		if (pMultiply != NULL)
//		{
//			// 调用 DLL 中的函数
//			//int result = pMultiply(5, 7);
//			pMultiply();
//
//			// 输出结果
//			//std::cout << "Result: " << result << std::endl;
//		}
//		else
//		{
//			// 获取函数地址失败
//			std::cerr << "Failed to get function address" << std::endl;
//		}
//
//		// 卸载 DLL
//		FreeLibrary(hinstDLL);
//	}
//	else
//	{
//		// 加载 DLL 失败
//		std::cerr << "Failed to load DLL" << std::endl;
//	}
//
//	// 等待用户输入并关闭程序
//	system("pause");
//	return 0;
//}
