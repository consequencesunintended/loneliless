#pragma once

#include "ofMain.h"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#define WIDTH_RES 160
#define HEIGHT_RES 160


namespace py = pybind11;

class ofApp : public ofBaseApp {

	public:
	void	setup();
	void	update();
	void	updateBallPosition( float dt, bool& retflag, bool& done );
	void	draw();
	bool	hasCollidedWithPlayer( const ofVec2f& ball_current_position, const ofVec2f& ball_new_position );

	void	moveUp( float dx );

	void	keyPressed( int key );
	void	moveDown( float dx );
	void	resetLevel();
	void	keyReleased( int key );
	void	mouseMoved( int x, int y );
	void	mouseDragged( int x, int y, int button );
	void	mousePressed( int x, int y, int button );
	void	mouseReleased( int x, int y, int button );
	void	mouseEntered( int x, int y );
	void	mouseExited( int x, int y );
	void	windowResized( int w, int h );
	void	dragEvent( ofDragInfo dragInfo );
	void	gotMessage( ofMessage msg );

	private:
	int				m_player_length{ HEIGHT_RES / 9 };
	ofVec2f			m_player_position{ 10.0f, 0.0f };
	ofVec2f			m_ball_position{ WIDTH_RES / 2, HEIGHT_RES / 2 };
	int				m_ball_size{ 5 };
	ofVec2f			m_ball_origin;
	ofVec2f			m_ball_direction{ 0.0f, -1.0f };
	ofVec2f			m_ball_original_direction;

	bool			m_key_pressed{ false };
	int				m_start_time{ 0 };
	int				m_end_time{ 0 };
	int				m_elapsed_time{ 0 };
	bool			m_initial_frames_set{ false };
	const int		m_num_of_frames_to_buffer{ 4 };

	py::module py_test;

	py::scoped_interpreter guard{};

	//ofTrueTypeFont 	vagRounded;
};
