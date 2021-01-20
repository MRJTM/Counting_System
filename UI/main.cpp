#include "countingtool.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    CountingTool w;
    w.show();

    return a.exec();
}
