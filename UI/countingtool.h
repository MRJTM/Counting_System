#ifndef COUNTINGTOOL_H
#define COUNTINGTOOL_H

#include <QMainWindow>

namespace Ui {
class CountingTool;
}

class CountingTool : public QMainWindow
{
    Q_OBJECT

public:
    explicit CountingTool(QWidget *parent = nullptr);
    ~CountingTool();

private:
    Ui::CountingTool *ui;
};

#endif // COUNTINGTOOL_H
