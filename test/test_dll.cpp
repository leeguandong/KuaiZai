#include <iostream>
#include <windows.h>

//// ����ָ�����͵Ķ��壬���ڱ��� DLL �к����ĵ�ַ
//typedef int (*START)();
//
//int main11()
//{
//	// ���� DLL
//	HINSTANCE hinstDLL = LoadLibrary(TEXT("KuaiZai.dll"));
//
//	// �ж� DLL �Ƿ���سɹ�
//	if (hinstDLL != NULL)
//	{
//		// ��ȡ DLL �к����ĵ�ַ
//		START pMultiply = (START)GetProcAddress(hinstDLL, "start");
//
//		// �жϺ�����ַ�Ƿ��ȡ�ɹ�
//		if (pMultiply != NULL)
//		{
//			// ���� DLL �еĺ���
//			//int result = pMultiply(5, 7);
//			pMultiply();
//
//			// ������
//			//std::cout << "Result: " << result << std::endl;
//		}
//		else
//		{
//			// ��ȡ������ַʧ��
//			std::cerr << "Failed to get function address" << std::endl;
//		}
//
//		// ж�� DLL
//		FreeLibrary(hinstDLL);
//	}
//	else
//	{
//		// ���� DLL ʧ��
//		std::cerr << "Failed to load DLL" << std::endl;
//	}
//
//	// �ȴ��û����벢�رճ���
//	system("pause");
//	return 0;
//}
